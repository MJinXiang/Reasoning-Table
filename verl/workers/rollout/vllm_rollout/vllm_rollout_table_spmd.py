"""
PythonVLLMRollout: A vLLM-based inference class that supports Python tool execution with a unified format.
Implements Python code execution using <python> and <output> tags.
Enhanced with improved memory management, multi-modal support, and tensor parallel processing.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import copy
import time
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf

from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length


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


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    """
    Repeat and interleave a tensor or numpy array
    
    Args:
        value: Tensor or array to repeat
        repeats: Number of repetitions
        
    Returns:
        Repeated and interleaved value
    """
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


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


def execute_with_dataframe(code, df):
    """
    Execute Python code with a DataFrame available in the namespace
    
    Args:
        code: Python code to execute
        df: Pandas DataFrame to make available to the code
        
    Returns:
        Tuple of (result, error_message)
    """
    # Create namespace with pandas and the dataframe
    namespace = {
        'pd': pd,  # Make pandas available in the code
        'df': df   # Provide the dataframe
    }
    
    try:
        exec(code, namespace)
        
        # Check for function definition
        function_name = next((name for name, obj in namespace.items() 
                          if callable(obj) and name != 'print' and not name.startswith('__')), None)
        
        if function_name:
            # Execute function with the DataFrame
            result = namespace[function_name](df)
            return result, None
        else:
            # Return last variable or expression result
            return namespace.get('__builtins__', {}).get('_', None), None
    except Exception as e:
        return None, f"Error executing code: {str(e)}"


class PythonVLLMRollout(BaseRollout):
    """
    vLLM-based rollout class that supports Python code execution
    Uses <python> and <output> tags for code execution
    Enhanced with multi-modal support and improved memory management
    """
    
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """
        Initialize the tool-enabled vLLM rollout
        
        Args:
            model_path: Path to the model
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
            if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
                train_tp = kwargs.get('train_tp', None)
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        # Validate model context length
        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
           
        # Calculate maximum model length 
        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')
        
        # Handle multimodal setting
        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format
        
        # Configure multimodal image limits if specified
        limit_mm_per_prompt = None
        if config.get('limit_images', None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get('limit_images')}
            
        # Initialize vLLM inference engine
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get('seed', 0),
        )

        # Use sleep mode to reduce memory usage
        self.inference_engine.sleep(level=1)

        # Set up basic sampling parameters
        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor recompute
            max_tokens=config.response_length,
        )

        # Handle vLLM version specifics
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # Add any custom sampling parameters from config
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        
        # Python code execution parameters
        self.max_action_iterations = config.get('max_action_iterations', 10)
        self.python_code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
        
        # Print configuration info
        print(f"Initialized PythonVLLMRollout with Python tool support")
        print(f"Max action iterations: {self.max_action_iterations}")
    
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

    def process_python_code(self, text):
        """
        Process text to find and execute Python code
        
        Args:
            text: Text potentially containing Python code blocks
            
        Returns:
            Tuple of (processed_text, has_python_code)
        """
        match = self.python_code_pattern.search(text)
        if not match:
            return text, False
            
        code = match.group(1)
        
        # Create a simple dummy DataFrame if needed
        # In a real application, you'd want to use the actual DataFrame from the context
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Execute the code with the DataFrame
        result, error = execute_with_dataframe(code, df)
        
        # Format the output
        if error:
            output = f"```output\n{error}\n```"
        else:
            output = f"```output\n{result}\n```"
            
        # Replace the code block with code + output
        processed_text = text[:match.end()] + "\n" + output
        
        return processed_text, True

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using vLLM with Python code execution support
        
        Args:
            prompts: Input prompts
            **kwargs: Additional parameters
            
        Returns:
            Generated sequences with processed Python code execution
        """
        # Rebuild vLLM cache engine if needed
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # Extract input tensors from prompts
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        # Handle non-tensor batch data (for multimodal inputs)
        non_tensor_batch = prompts.non_tensor_batch if hasattr(prompts, 'non_tensor_batch') else {}
        
        # Process raw prompt ids for efficient vLLM inference
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)
            
        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vLLM sharding manager is not working properly.')
            
        # Prepare vLLM inputs with multi-modal support if available
        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]
            
        # Ensure prompt_token_ids are correctly formatted for vLLM
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        # Get sampling settings
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        
        # Set up sampling parameters based on validation or greedy settings
        sampling_kwargs = {}
        if not do_sample:
            sampling_kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            sampling_kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # Generate initial responses
        start_time = time.time()
        with self.update_sampling_params(**sampling_kwargs):
            # Create temporary sampling params for Python code processing
            tmp_sampling_params = copy.deepcopy(self.sampling_params)
            tmp_sampling_params.max_tokens = 2048
            safe_set_stop_tokens(tmp_sampling_params, eos_token_id)
            
            # Perform generation
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=tmp_sampling_params,
                use_tqdm=True)
            
            # Process outputs from vLLM
            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    
            # Pad responses to consistent length
            response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)
        print(f"Initial generation time: {time.time() - start_time:.2f}s")

        # Decode responses for Python code processing
        current_prefix_list = []
        for i in range(batch_size):
            current_prefix_list.append(self.tokenizer.decode(
                _pre_process_inputs(self.pad_token_id, idx[i]), 
                skip_special_tokens=False
            ))
        
        response_str_list = [
            self.tokenizer.decode(response[i].tolist(), skip_special_tokens=False) 
            for i in range(response.size(0))
        ]
        raw_current_prefix_list = copy.deepcopy(current_prefix_list)
        
        # Set up parameters for continued generation after code execution
        re_sampling_params = copy.deepcopy(self.sampling_params)
        re_sampling_params.n = 1
        re_sampling_params.max_tokens = 1024
        safe_set_stop_tokens(re_sampling_params, eos_token_id)

        # Python code execution iteration loop
        pber = tqdm(range(self.max_action_iterations), desc="Processing Python code blocks...", disable=False)
        for iter in pber:
            # Validate list lengths
            assert len(response_str_list) == len(current_prefix_list), \
                "response_str_list and current_prefix_list should have the same length"
                
            start_time = time.time()
            
            # Initialize tracking for this iteration
            new_prefix_list = current_prefix_list.copy()
            action_flag_list = [False] * len(current_prefix_list)
            
            # Process each response for Python code blocks
            for i, response_str in enumerate(response_str_list):
                processed_text, has_python_code = self.process_python_code(response_str)
                
                if has_python_code:
                    # Update prefix with code execution results
                    new_prefix_list[i] = current_prefix_list[i] + processed_text
                    action_flag_list[i] = True
            
            # Update prefixes for next iteration
            current_prefix_list = new_prefix_list
            
            # Update progress bar description
            pber.set_description(
                f"Python execution iteration {iter+1}, time: {time.time() - start_time:.2f}s, "
                f"code executed: {sum(action_flag_list)}/{len(action_flag_list)}"
            )
            
            # Continue generation if Python code blocks were executed
            if any(action_flag_list):
                # Collect prompts that need continued generation
                new_prompts_list = []
                new_prompt_indices = []
                
                for i, action_flag in enumerate(action_flag_list):
                    if action_flag:
                        new_prompts_list.append(current_prefix_list[i])
                        new_prompt_indices.append(i)
                    
                if new_prompts_list:
                    # Encode the new prompts
                    input_ids_list = []
                    for p in new_prompts_list:
                        input_ids_list.append(self.tokenizer.encode(p, add_special_tokens=False))
                    
                    # Generate continuations
                    new_outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=re_sampling_params,
                        prompt_token_ids=input_ids_list,
                        use_tqdm=False
                    )
                    
                    # Extract token IDs from outputs
                    new_responses = []
                    for output in new_outputs:
                        for sample_id in range(len(output.outputs)):
                            new_responses.append(output.outputs[sample_id].token_ids)
                    
                    # Decode the new responses
                    new_response_str_list = [
                        self.tokenizer.decode(new_response, skip_special_tokens=False)
                        for new_response in new_responses
                    ]
                    
                    # Update responses
                    for idx, i in enumerate(new_prompt_indices):
                        response_str_list[i] = new_response_str_list[idx]
            else:
                # Exit loop if no Python code blocks detected
                break
        
        pber.close()
        
        # Convert processed responses back to token IDs
        final_responses = []
        for i in range(len(response_str_list)):
            # Extract only the response part (without prefix)
            response_only = response_str_list[i]
            tokens = self.tokenizer.encode(response_only, add_special_tokens=False)
            
            # Ensure we don't exceed max length
            tokens = tokens[:self.config.response_length]
            
            # Add EOS token if needed
            if len(tokens) < self.config.response_length:
                tokens = tokens + [eos_token_id]
                
            # Pad to full length
            while len(tokens) < self.config.response_length:
                tokens.append(self.pad_token_id)
                
            final_responses.append(tokens)
        
        # Convert to tensor
        response = torch.tensor(final_responses, device=idx.device)
        
        # Handle multiple samples for n > 1
        if self.sampling_params.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.sampling_params.n)
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            batch_size = batch_size * self.sampling_params.n
            if 'multi_modal_inputs' in non_tensor_batch.keys():
                non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                            self.sampling_params.n)
            
        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Build position IDs for response
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        # Handle multi-dimensional position IDs (e.g., for rotary position embeddings)
        if position_ids.dim() == 3:  # qwen2vl mrope or similar architectures
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        
        # Combine position IDs
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Build attention mask for response
        response_attention_mask = get_response_mask(response_id=response,
                                                  eos_token=eos_token_id,
                                                  dtype=attention_mask.dtype)
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
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)