"""
ToolsVLLMRollout: A vLLM-based inference class that supports tool calling with a unified format.
Implements tool calling using <tool_call> and <observation> tags instead of tool-specific tags.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import copy
import time
import re
import json
from tqdm import tqdm
from contextlib import contextmanager

from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length

# Import tool-related modules
from verl.tools.tool_manager import ToolManager
from verl.tools.calculator import CalculatorTool
from verl.tools.python_executor import PythonExecutorTool


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """
    Remove left padding from prompt token IDs
    
    Args:
        pad_token_id: The ID of padding token
        prompt_token_ids: Tensor containing prompt token IDs
        
    Returns:
        List of token IDs with padding removed
    """
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def safe_set_stop_tokens(sampling_params, tokens):
    """
    Safely set stop tokens for sampling parameters, avoiding nested list issues
    
    Args:
        sampling_params: vLLM sampling parameters
        tokens: Token IDs to stop generation (can be int, list, or nested list)
    """
    if tokens is None:
        sampling_params.stop_token_ids = None
        return
            
    flat_tokens = []
    if isinstance(tokens, (list, tuple)):
        for item in tokens:
            if isinstance(item, (list, tuple)):
                flat_tokens.extend([int(t) for t in item if t is not None])
            elif item is not None:
                flat_tokens.append(int(item))
    else:
        flat_tokens = [int(tokens)] if tokens is not None else []
            
    sampling_params.stop_token_ids = flat_tokens


class ToolsVLLMRollout(BaseRollout):
    """
    vLLM-based rollout class that supports generic tool calling
    Uses <tool_call> and <observation> tags for tool execution
    """
    
    def __init__(self, actor_module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """
        Initialize the tool-enabled vLLM rollout
        
        Args:
            actor_module: The model module following HuggingFace API
            config: Configuration dict
            tokenizer: The tokenizer for the model
            model_hf_config: HuggingFace config for model initialization
            **kwargs: Additional arguments for Megatron backend
        """
        super().__init__()
        self.config = config
        
        # Validate configuration
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        # Get tensor parallel configuration
        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        # Handle Megatron backend if used
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        # Validate model context length
        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
            
        # Initialize vLLM inference engine
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload model weights to reduce peak memory
        self.inference_engine.offload_model_weights()

        # Set up basic sampling parameters
        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor recompute
            max_tokens=config.response_length,
        )

        # Handle vLLM version specifics
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # Add any custom sampling parameters from config
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        
        # Tool calling related parameters
        self.max_action_iterations = config.get('max_action_iterations', 10)
        
        # Initialize tool manager and register tools
        self.tool_manager = ToolManager()
        
        # Register tools based on configuration
        enabled_tools = self._get_enabled_tools_from_config(config)
        
        if 'calculator' in enabled_tools:
            self.tool_manager.register_tool(CalculatorTool())
            
        if 'python_executor' in enabled_tools:
            self.tool_manager.register_tool(PythonExecutorTool())
        
        # Log enabled tools
        print(f"Enabled tools: {enabled_tools}")
    
    def _get_enabled_tools_from_config(self, config):
        """
        Extract enabled tools from configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of enabled tool names
        """
        all_available_tools = ['calculator', 'python_executor']
        
        if 'enabled_tools' in config:
            tools = config.get('enabled_tools')
            if isinstance(tools, str):
                return tools.split(',')
            elif isinstance(tools, list):
                return tools
            else:
                return all_available_tools
        else:
            return all_available_tools
    
    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        Context manager to temporarily update sampling parameters
        
        Args:
            **kwargs: Parameters to update
        """
        # Save old values
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # Restore old values
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using vLLM with tool support
        
        Args:
            prompts: Input prompts
            **kwargs: Additional parameters
            
        Returns:
            Generated sequences with processed tool calls
        """
        # Rebuild vLLM cache engine if needed
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # Extract input tensors from prompts
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        # Convert input tensors to token lists
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        # Get sampling settings
        do_sample = prompts.meta_info.get('do_sample', True)
        
        # Set up greedy decoding if requested
        greedy_kwargs = {}
        if not do_sample:
            greedy_kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # Generate initial responses
        start_time = time.time()
        with self.update_sampling_params(**greedy_kwargs):
            current_n = self.sampling_params.n
            # Create temporary sampling params with longer max_tokens
            tmp_sampling_params = copy.deepcopy(self.sampling_params)
            tmp_sampling_params.max_tokens = 2048
            safe_set_stop_tokens(tmp_sampling_params, eos_token_id)
            
            # Perform initial generation
            output = self.inference_engine.generate(
                prompts=None,
                sampling_params=tmp_sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=True)
        print(f"Initial generation time: {time.time() - start_time:.2f}s")

        # Extract generated responses
        response = output[0].tolist()
        
        # Prepare for tool processing iterations
        current_prefix_list = []
        for sample_idx in range(len(idx_list)):
            for _n in range(current_n):
                current_prefix_list.append(idx_list[sample_idx])
        
        # Decode current prefixes and responses to text
        current_prefix_list = self.tokenizer.batch_decode(current_prefix_list, skip_special_tokens=False)
        response_str_list = self.tokenizer.batch_decode(response, skip_special_tokens=False)
        raw_current_prefix_list = copy.deepcopy(current_prefix_list)
        
        # Set up parameters for continued generation after tool calls
        re_sampling_params = copy.deepcopy(self.sampling_params)
        re_sampling_params.n = 1
        re_sampling_params.max_tokens = 1024
        safe_set_stop_tokens(re_sampling_params, eos_token_id)

        # Tool calling iteration loop
        pber = tqdm(range(self.max_action_iterations + 1), desc="Processing tool calls...", disable=False)
        for iter in pber:
            # Validate list lengths
            assert len(response_str_list) == len(current_prefix_list), \
                "response_str_list and current_prefix_list should have the same length"
            
            # Extend max tokens for final iteration
            if iter == self.max_action_iterations:
                re_sampling_params.max_tokens = 2048
                
            start_time = time.time()
            
            # Initialize tracking for this iteration
            new_prefix_list = current_prefix_list.copy()
            action_flag_list = [False] * len(current_prefix_list)
            
            # Process each response for tool calls
            for i, response_str in enumerate(response_str_list):
                has_tool_call = "<tool_call>" in response_str
                if has_tool_call:
                    print(f"DEBUG: Sample {i} contains tool call:")
                    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', response_str, re.DOTALL)
                    if tool_call_match:
                        print(f"  Tool call content: {tool_call_match.group(1)[:100]}...")
    
                processed_text, has_tool_call = self.tool_manager.process_text(response_str)
                
                if has_tool_call:
                    observation_added = "<observation>" in processed_text
                    print(f"DEBUG: Tool call processed for sample {i}, observation added: {observation_added}")
                    # Update prefix with tool call results
                    new_prefix_list[i] = current_prefix_list[i] + processed_text
                    action_flag_list[i] = True
            
            # Log debug information for tool calls
            if any(action_flag_list):
                print(f"DEBUG - Tool call iteration {iter+1}:")
                for i, (flag, prefix) in enumerate(zip(action_flag_list, new_prefix_list)):
                    if flag:
                        print(f"Sample {i} detected tool call, new prefix length: {len(prefix)}")
                        print(f"Prefix end: {prefix[-100:] if len(prefix) > 100 else prefix}")

            # Update prefixes for next iteration
            current_prefix_list = new_prefix_list
            
            # Update progress bar description
            pber.set_description(
                f"Tool iteration {iter+1}, time: {time.time() - start_time:.2f}s, "
                f"tools detected: {sum(action_flag_list)}/{len(action_flag_list)}"
            )
            
            # Continue generation if tool calls were detected
            if any(action_flag_list):
                # Collect prompts that need continued generation
                new_prompts_list = []
                for action_flag, current_prefix in zip(action_flag_list, current_prefix_list):
                    if action_flag:
                        new_prompts_list.append(current_prefix)
                    
                if new_prompts_list:
                    # Encode the new prompts
                    input_ids_list = []
                    for p in new_prompts_list:
                        input_ids_list.append(self.tokenizer.encode(p, add_special_tokens=False))
                    
                    # Generate continuations
                    new_response_list = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=re_sampling_params,
                        prompt_token_ids=input_ids_list,
                        use_tqdm=False
                    )[0]
                    
                    # Decode the new responses
                    new_response_str_list = self.tokenizer.batch_decode(new_response_list, skip_special_tokens=False)
                    
                    # Update responses
                    j = 0
                    for i, action_flag in enumerate(action_flag_list):
                        if action_flag:
                            response_str_list[i] = new_response_str_list[j]
                            j += 1
                            
                    # Log detected tool calls
                    if any(action_flag_list):
                        print("Detected tool calls:")
                        for i, flag in enumerate(action_flag_list):
                            if flag:
                                print(f"Sample {i}: Called tool, response: {response_str_list[i][:100]}...")
            else:
                # Exit loop if no tool calls detected
                break
            
        pber.close()
        
        # Construct full content by combining prefixes and responses
        full_content = []
        for response_str, current_prefix in zip(response_str_list, current_prefix_list):
            full_content.append(current_prefix + response_str)
        
        # Extract the response part (without the original prompt)
        response = []
        for p_id, p in enumerate(full_content):
            response.append(self.tokenizer.encode(p[len(raw_current_prefix_list[p_id]):], add_special_tokens=False))

        # Find maximum response length and handle EOS tokens
        max_response_len = -1
        for i_res in range(len(response)):
            # Remove consecutive EOS tokens
            while len(response[i_res]) > 1 and response[i_res][-1] == eos_token_id:
                if response[i_res][-2] == eos_token_id:
                    response[i_res] = response[i_res][:-1]
                else:
                    break
            
            max_response_len = max(max_response_len, len(response[i_res]))

        # Log debug information about responses
        print(f"DEBUG - response type: {type(response)}")
        if isinstance(response, list) and len(response) > 0:
            print(f"DEBUG - first element type: {type(response[0])}")
            print(f"DEBUG - first element value: {response[0][:10] if len(response[0]) > 10 else response[0]}")

        # Pad responses to the same length and apply max_tokens constraint
        with self.update_sampling_params(**greedy_kwargs):
            for i_res in range(len(response)):
                # Add EOS padding
                response[i_res] = response[i_res] + [eos_token_id] * (max_response_len - len(response[i_res]))
                # Truncate if longer than max_tokens
                response[i_res] = response[i_res][:self.sampling_params.max_tokens]

        # Handle potential nested lists in response
        if isinstance(response, list):
            processed_response = []
            for i_res in range(len(response)):
                if isinstance(response[i_res], list):
                    # Process nested lists
                    current_res = []
                    for item in response[i_res]:
                        if item is None:
                            current_res.append(0)
                        elif isinstance(item, list):
                            # Take first valid value from nested list
                            if item and len(item) > 0:
                                current_res.append(int(item[0]) if item[0] is not None else 0)
                            else:
                                current_res.append(0)
                        else:
                            # Normal case
                            current_res.append(int(item))
                    processed_response.append(current_res)
                else:
                    # Convert single item to list
                    processed_response.append([int(response[i_res]) if response[i_res] is not None else 0])
            response = processed_response

        # Convert processed list to tensor
        response = torch.tensor(response, device=attention_mask.device, dtype=attention_mask.dtype)

        # Pad response if shorter than configured length
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)

        # Handle multiple samples for n > 1
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            
        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Build position IDs for response
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Build attention mask for response
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Create batch tensor with all results
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # full sequences (prompt + response)
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # Free vLLM cache engine if configured
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)