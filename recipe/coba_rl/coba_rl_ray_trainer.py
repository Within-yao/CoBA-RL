import os
import uuid
import numpy as np
import torch
import ray
from collections import defaultdict
from pprint import pprint

from tqdm import tqdm
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_response_mask,
    compute_advantage,
    apply_kl_penalty
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
from verl.utils.metric import reduce_metrics
from verl.utils.debug import marked_timer
import json




class RayCobaRLTrainer(RayPPOTrainer):
    """
    CoBA-RL (Constraint-based Budget Allocation for Reinforcement Learning) Trainer
    
    Extends RayPPOTrainer with dynamic budget allocation for sample rollouts.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize budget allocator
        allocator_type = self.config.trainer.get("allocator_type", "beta")
        allocator_config = self.config.trainer.get("allocator_config", {})
        
        
        default_n = self.config.actor_rollout_ref.rollout.n
        train_batch_size = self.config.data.train_batch_size
        total_rollout_budget = train_batch_size * default_n
        
        
        allocator_config = dict(allocator_config)  
        allocator_config['total_rollout_budget'] = total_rollout_budget
        
       
        total_epochs = self.config.trainer.total_epochs
        steps_per_epoch = len(self.train_dataloader)
        total_training_steps = total_epochs * steps_per_epoch
        allocator_config['total_training_steps'] = total_training_steps
        
        print(f"Total training steps: {total_training_steps} (epochs={total_epochs}, steps_per_epoch={steps_per_epoch})")
        
        if allocator_type == "beta":
            from recipe.coba_rl.budget_allocators import BetaAllocator
            self.budget_allocator = BetaAllocator(**allocator_config)
        else:
            raise ValueError(f"Unknown allocator_type: {allocator_type}. Only 'beta' is supported.")
        
        print(f"Initialized {allocator_type} allocator with config: {allocator_config}")

    
    def fit(self):
        """
        The training loop of CoBA-RL.
        Incorporates dynamic budget allocation based on sample success rates.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Dynamically compute save_freq based on save_times_per_epoch
        if self.config.trainer.get("save_times_per_epoch", 0) > 0:
            steps_per_epoch = len(self.train_dataloader)
            save_times = self.config.trainer.save_times_per_epoch
            self.config.trainer.save_freq = max(1, steps_per_epoch // save_times)
            self.config.trainer.test_freq = self.config.trainer.save_freq
            
            print(f"Overriding save_freq & test_freq to {self.config.trainer.save_freq} "
                  f"because save_times_per_epoch={save_times} (steps_per_epoch={steps_per_epoch})")

        self.global_steps += 1
        last_val_metrics = None
        default_n = self.config.actor_rollout_ref.rollout.n
        train_batch_size = self.config.data.train_batch_size
        total_rollout_budget = train_batch_size * default_n

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = defaultdict(float)

                # Prepare batch
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                global_indices = batch.non_tensor_batch["sample_global_index"].astype(int)

                # Budget allocation
                with marked_timer("budget_allocation", timing_raw):
                    # Update allocator current training step (for step_based_linear)
                    if hasattr(self.budget_allocator, 'set_current_training_step'):
                        self.budget_allocator.set_current_training_step(self.global_steps)
                    
                    # Allocate budget using allocator
                    allocation_counts = self.budget_allocator.allocate(
                        global_indices=global_indices,
                        total_budget=total_rollout_budget
                    )
                    
                    # Print allocation results
                    allocator_type = type(self.budget_allocator).__name__
                    print("\n" + "="*80)
                    print(f"[Step {self.global_steps}] Budget Allocation Result ({allocator_type})")
                    print("="*80)
                    print(f"Total Budget: {total_rollout_budget}")
                    print(f"Sample Count: {len(global_indices)}")
                    print(f"Actual Allocated: {np.sum(allocation_counts)}")
                    print("-"*80)
                    
                    # Get current success rates (from allocator's task_stats)
                    if hasattr(self.budget_allocator, 'task_stats'):
                        # Check if any sample has history_budget
                        has_history_budget = False
                        for gidx in global_indices:
                            stats = self.budget_allocator.task_stats.get(int(gidx), {})
                            if stats.get("history_budget", 0) > 0:
                                has_history_budget = True
                                break
                        
                        # Choose print format based on history_budget existence
                        if has_history_budget:
                            print(f"{'Index':<8} {'Global Idx':<12} {'Success Rate':<12} {'History Budget':<12} {'Allocated':<10}")
                            print("-"*80)
                            for i, (gidx, count) in enumerate(zip(global_indices, allocation_counts)):
                                stats = self.budget_allocator.task_stats.get(int(gidx), {})
                                srate = stats.get("success_rate", None)
                                srate_str = "none" if srate is None else f"{srate:.4f}"
                                history_budget = stats.get("history_budget", 0)
                                print(f"{i:<8} {gidx:<12} {srate_str:<12} {history_budget:<12} {count:<10}")
                        else:
                            print(f"{'Index':<8} {'Global Idx':<12} {'Success Rate':<12} {'Allocated':<10}")
                            print("-"*80)
                            for i, (gidx, count) in enumerate(zip(global_indices, allocation_counts)):
                                stats = self.budget_allocator.task_stats.get(int(gidx), {})
                                srate = stats.get("success_rate", None)
                                srate_str = "none" if srate is None else f"{srate:.4f}"
                                print(f"{i:<8} {gidx:<12} {srate_str:<12} {count:<10}")
                    else:
                        print(f"{'Index':<8} {'Global Idx':<12} {'Allocated':<10}")
                        print("-"*80)
                        for i, (gidx, count) in enumerate(zip(global_indices, allocation_counts)):
                            print(f"{i:<8} {gidx:<12} {count:<10}")
                    
                    # Record allocation statistics
                    metrics["allocation/mean_rollouts"] = np.mean(allocation_counts)
                    metrics["allocation/max_rollouts"] = np.max(allocation_counts)
                    metrics["allocation/min_rollouts"] = np.min(allocation_counts)
                    metrics["allocation/active_samples"] = np.sum(allocation_counts > 0)
                    metrics["allocation/zero_rollouts_cnt"] = np.sum(allocation_counts == 0)

                    # Construct batch based on allocation
                    repeat_indices = []
                    for idx, count in enumerate(allocation_counts):
                        if count > 0:
                            repeat_indices.extend([idx] * count)
                    
                    if not repeat_indices:
                        print("Warning: All samples allocated 0 rollouts.")
                        continue
                    
                    batch = batch[repeat_indices]
                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps
                # Rollout generation
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # Update allocator statistics
                        with marked_timer("update_allocator_stats", timing_raw):
                            current_indices = batch.non_tensor_batch["sample_global_index"].astype(int)
                            seq_scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()

                            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                            if rollout_data_dir is not None:
                                acc_log_path = os.path.join(rollout_data_dir, f"training_acc_{self.global_steps}.jsonl")
                                
                                try:
                                    with open(acc_log_path, 'w') as f:
                                        for idx, score in zip(current_indices, seq_scores):
                                            record = {
                                                "sample_global_index": int(idx),
                                                "acc": float(score)
                                            }
                                            f.write(json.dumps(record) + "\n")
                                except Exception as e:
                                    print(f"Error logging accuracy: {e}")
                            
                            # Update success rate statistics
                            self.budget_allocator.update_stats(
                                global_indices=current_indices,
                                rewards=seq_scores
                            )
                            
                            # Commit step statistics (for sliding window/Beta mode)
                            self.budget_allocator.commit_step_stats()
                            
                            # Log updated beta_alpha value (if BetaAllocator)
                            if hasattr(self.budget_allocator, 'beta_alpha'):
                                metrics["beta/beta_alpha"] = float(self.budget_allocator.beta_alpha)
                                if hasattr(self.budget_allocator, 'beta_beta'):
                                    metrics["beta/beta_beta"] = float(self.budget_allocator.beta_beta)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=1, # NOTE: 传1
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Update Actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        progress_bar.close()

    def _save_checkpoint(self):
        super()._save_checkpoint()

        # Save allocator state
        allocator_state_path = os.path.join(
            self.config.trainer.default_local_dir, 
            f"global_step_{self.global_steps}", 
            "allocator_state.json"
        )
        self.budget_allocator.save_stats(allocator_state_path)

    def _load_checkpoint(self):
        res = super()._load_checkpoint()

        # Load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # Try to load allocator state
        if self.config.trainer.resume_mode != "disable":
            if global_step_folder and os.path.exists(os.path.join(global_step_folder, "allocator_state.json")):
                allocator_state_path = os.path.join(global_step_folder, "allocator_state.json")
                # Decide whether to restore training state based on resume_mode
                # "all" or "auto": restore full training state (current_step, window data, etc.)
                # "model": only load prior success rates, reset training state
                resume = (self.config.trainer.resume_mode in ["all", "auto"])
                self.budget_allocator.load_stats(allocator_state_path, resume=resume)
                print(f"[CoBA-RL] Loaded allocator stats from {allocator_state_path} (resume={resume}, mode={self.config.trainer.resume_mode})")
            else:
                print(f"[CoBA-RL] No allocator_state.json found in {global_step_folder or 'None'}")
        return res