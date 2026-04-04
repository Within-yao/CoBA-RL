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

from dataclasses import is_dataclass
from typing import Any, Optional
import os

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config", "print_experiment_summary"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        assert config.actor_rollout_ref.model.lora_rank <= 512, "LoRA rank in vLLM must be less than or equal to 512"

    print("[validate_config] All configuration checks passed successfully!")

def print_experiment_summary(config):
    """
    Print a summary table of the experiment configuration and hyperparameters.
    Useful for copying to experiment tracking sheets.
    """
    try:
        # Helper function to safely get nested config values
        def get_cfg(cfg, path, default="N/A"):
            try:
                val = cfg
                for key in path.split('.'):
                    val = val[key]
                return val
            except Exception:
                return default

        # 1. Extract Basic Info
        exp_name = config.trainer.get("experiment_name", "")
        time_str = exp_name.split('_')[-1] if '_' in exp_name else "N/A"
        
        model_path = get_cfg(config, "actor_rollout_ref.model.path", "")
        model_name = os.path.basename(model_path.rstrip('/')) if model_path else "N/A"

        # 2. Build Summary Dictionary
        summary_config = {
            # --- Identity ---
            "time_str": time_str,
            "model": model_name,
            "adv_estimator": config.algorithm.get("adv_estimator", "N/A"),
            "total_epochs": config.trainer.get("total_epochs", "N/A"),
            
            # --- Hyperparameters ---
            "lr": get_cfg(config, "actor_rollout_ref.actor.optim.lr"),
            "ppo_mini_batch_size": get_cfg(config, "actor_rollout_ref.actor.ppo_mini_batch_size"),
            "ppo_micro_batch_size_per_gpu": get_cfg(config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"),
            "kl_loss_coef": get_cfg(config, "actor_rollout_ref.actor.kl_loss_coef"),
            "kl_loss_type": get_cfg(config, "actor_rollout_ref.actor.kl_loss_type"),
            "entropy_coeff": get_cfg(config, "actor_rollout_ref.actor.entropy_coeff"),
            "grad_clip": get_cfg(config, "actor_rollout_ref.actor.grad_clip", "N/A"),
            
            # --- Data ---
            "train_batch_size": config.data.get("train_batch_size", "N/A"),
            "max_prompt_length": config.data.get("max_prompt_length", "N/A"),
            "max_response_length": config.data.get("max_response_length", "N/A"),
            
            # --- Rollout ---
            "rollout_n": get_cfg(config, "actor_rollout_ref.rollout.n"),
            "rollout_tp": get_cfg(config, "actor_rollout_ref.rollout.tensor_model_parallel_size"),
            "rollout_temperature": get_cfg(config, "actor_rollout_ref.rollout.temperature", "N/A"),
            
            # --- Resources ---
            "nnodes": config.trainer.get("nnodes", "N/A"),
            "n_gpus_per_node": config.trainer.get("n_gpus_per_node", "N/A"),
            "save_freq": config.trainer.get("save_freq", "N/A"),
            "save_times_per_epoch": config.trainer.get("save_times_per_epoch", "N/A"),
            "val_before_train": config.trainer.get("val_before_train", "N/A"),
            "auto_merge_checkpoints": config.trainer.get("auto_merge_checkpoints", "N/A"),
            
            # --- Allocator ---
            "allocator_type": config.trainer.get("allocator_type", "N/A"),
            "allocator_n_low": get_cfg(config, "trainer.allocator_config.n_low", "N/A"),
            "allocator_n_up": get_cfg(config, "trainer.allocator_config.n_up", "N/A"),
            "allocator_update_method": get_cfg(config, "trainer.allocator_config.update_method", "N/A"),
            "allocator_beta_alpha_semantic": get_cfg(config, "trainer.allocator_config.beta_alpha_semantic", "N/A"),
            "allocator_sliding_window_size": get_cfg(config, "trainer.allocator_config.sliding_window_size", "N/A"),
            "allocator_enable_alpha_variance_adjustment": get_cfg(config, "trainer.allocator_config.enable_alpha_variance_adjustment", "N/A"),
            "allocator_update_mode": get_cfg(config, "trainer.allocator_config.update_mode", "N/A"),
            "allocator_smoothing_factor": get_cfg(config, "trainer.allocator_config.smoothing_factor", "N/A"),
            "allocator_tau_min": get_cfg(config, "trainer.allocator_config.tau_min", "N/A"),
            "allocator_tau_max": get_cfg(config, "trainer.allocator_config.tau_max", "N/A"),
            "allocator_var_max": get_cfg(config, "trainer.allocator_config.var_max", "N/A"),
            "allocator_exploration_budget_ratio": get_cfg(config, "trainer.allocator_config.exploration_budget_ratio", "N/A"),
            "allocator_remaining_budget_method": get_cfg(config, "trainer.allocator_config.remaining_budget_method", "N/A"),
            "allocator_enable_ucb": get_cfg(config, "trainer.allocator_config.enable_ucb", "N/A"),
            "allocator_ucb_scale": get_cfg(config, "trainer.allocator_config.ucb_scale", "N/A"),
            "allocator_use_heap_greedy": get_cfg(config, "trainer.allocator_config.use_heap_greedy", "N/A")
        }

        # 3. Print
        print("\n" + "="*20 + " EXPERIMENT SUMMARY " + "="*20)
        for key, value in summary_config.items():
            if value != "N/A":
                print(f"{key}\t{value}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[Warning] Failed to print experiment summary: {e}")