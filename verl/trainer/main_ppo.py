# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import os
import ray
import hydra


from verl.utils.reward_score import wikisql, wikitq, tatqa, hybridqa, tabfact, multihiertt, hitab, feverous, totto, fetaqa, gsm8k, finqa, ottqa  #gsm8k, math, multiply, countdown,tablebench,
from verl.utils.reward_score import table_evidence
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from tqdm import tqdm
import torch 
from verl import DataProto

def _select_rm_score_fn(data_source):
    if "wikisql" in data_source:
        return wikisql.compute_score
    elif "wikitq" in data_source:
        return wikitq.compute_score
    elif "tatqa" in data_source:
        return tatqa.compute_score
    elif "hybridqa" in data_source:
        return hybridqa.compute_score
    elif "tabfact" in data_source:
        return tabfact.compute_score
    elif "multihiertt" in data_source:
        return multihiertt.compute_score
    elif "hitab" in data_source:
        return hitab.compute_score
    elif "feverous" in data_source:
        return feverous.compute_score
    elif "totto" in data_source:
        return totto.compute_score
    elif "fetaqa" in data_source:
        return fetaqa.compute_score
    elif "gsm8k" in data_source:
        return gsm8k.compute_score
    elif "finqa" in data_source:
        return finqa.compute_score
    elif "ottqa" in data_source:
        return ottqa.compute_score
    
    else:
        raise NotImplementedError



class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, test_only=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.test_only = test_only  # parameter to control whether to return test scores only


    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # If test scores are needed, create an additional tensor
        if self.test_only:
            test_metrics = {
                'combined_scores': [],  # combined scores
                'em_scores': [],        # EM scores
                'f1_scores': [],        # F1 scores
                'bleu_scores': [],      # BLEU scores
                'accuracy_scores': [],  # accuracy scores
            }
    
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids)
            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # additional info
            extra_info = data_item.non_tensor_batch['extra_info']

            # default EM and F1 values set to 0
            em_score = 0
            f1_score = 0
            bleu_score = 0
            accuracy_score = 0

            if "evidence" in extra_info or (data_source =="tatqa" and "evidence" not in extra_info):
                assert data_source in ["hitab","tatqa","wikitq"], f"{data_source} is not supported for evidence"
                if self.test_only:
                    result = table_evidence.compute_score(data_source=data_source, solution_str=response_str, ground_truth=ground_truth, evidence=[], return_details=True)
                    if isinstance(result, dict):
                        score = result.get('combined_score', 0)
                        em_score = result.get('em', 0)
                        f1_score = result.get('f1', 0)
                        bleu_score = result.get('bleu', 0)
                        accuracy_score = result.get('accuracy', 0)
                    else:
                        score = result
                else:
                    score = table_evidence.compute_score(data_source=data_source, solution_str=response_str, ground_truth=ground_truth, evidence=extra_info['evidence'], return_details=False)
            elif data_source == 'wikisql':
                # additional
                table = data_item.non_tensor_batch['extra_info']['table']
                ans = data_item.non_tensor_batch['extra_info']['answer']
                if self.test_only:
                    result = compute_score_fn(solution_str=sequences_str,  table=table, ans=ans, ground_truth=ground_truth, return_details=True)
                    
                    if isinstance(result, dict):
                        score = result.get('combined_score', 0)
                        accuracy_score = result.get('accuracy', 0)
                    else:
                        score = result
                else:
                    score = compute_score_fn(solution_str=sequences_str, table=table, ans=ans, ground_truth=ground_truth)
            elif data_source == 'tatqa':
                extra_info = data_item.non_tensor_batch['extra_info']
                if self.test_only:
                    # For test mode, get detailed metrics
                    result = compute_score_fn(solution_str=sequences_str,
                        ground_truth=ground_truth, extra_info=extra_info, return_details=True  # Please ensure the scoring function supports this parameter
                    )
                    
                    # If the return is a detailed metrics dictionary
                    if isinstance(result, dict):
                        score = result.get('combined_score', 0)
                        em_score = result.get('em', 0)
                        f1_score = result.get('f1', 0)
                    else:
                        score = result
                else:
                    # Training mode, only need to return total score
                    score = compute_score_fn(solution_str=sequences_str,ground_truth=ground_truth, extra_info=extra_info)
                # For Tatqa, we need to pass the table and answer to the scoring function
                # score = compute_score_fn(solution_str=sequences_str, table=table, paragraphs=paragraphs, ground_truth=ground_truth, extra_info=extra_info)

            elif data_source in ['fetaqa', 'feverous', 'gsm8k', 'hitab', 'hybridqa','multihiertt','tabfact','totto', 'wikitq','finqa','ottqa']:
                if self.test_only:
                    result = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, extra_info=extra_info, return_details=True)
                    if isinstance(result, dict):
                        score = result.get('combined_score', 0)
                        em_score = result.get('em', 0)
                        f1_score = result.get('f1', 0)
                        bleu_score = result.get('bleu', 0)
                        accuracy_score = result.get('accuracy', 0)
                        has_answer = result.get('has_answer', 0)
                    else:
                        score = result
                else:
                    score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, extra_info=extra_info)
            else:  
                raise ValueError(f"Unsupported data source: {data_source}")

            reward_tensor[i, valid_response_length - 1] = score

            # Collect test metrics
            if self.test_only:
                test_metrics['combined_scores'].append(score)
                test_metrics['em_scores'].append(em_score)
                test_metrics['f1_scores'].append(f1_score)
                test_metrics['bleu_scores'].append(bleu_score)
                test_metrics['accuracy_scores'].append(accuracy_score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
                if self.test_only:
                    if 'fetaqa' in data_source or 'totto' in data_source:
                        print(f"Score: {score}, BLEU: {bleu_score}")
                    elif 'feverous' in data_source or 'gsm8k' in data_source or 'hitab' in data_source or 'tabfact' in data_source or 'wikisql' in data_source or 'wikitq' in data_source or 'finqa' in data_source: 
                        print(f"Score: {score}, Accuracy: {accuracy_score}")
                    else:
                        print(f"Score: {score}, EM: {em_score}, F1: {f1_score}")
        # If in pure test mode, save test scores to non_tensor_batch
        if self.test_only:
            data.non_tensor_batch['test_metrics'] = test_metrics

        return reward_tensor

import ray
import hydra

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

   


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        #use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = RewardManager(
            tokenizer=tokenizer, 
            num_examine=0
        )

        val_reward_fn = RewardManager(
            tokenizer=tokenizer, 
            num_examine=1, 
            test_only=True
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()