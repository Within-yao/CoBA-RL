"""
Budget Allocators for CoBA-RL Training

Implements BetaAllocator strategy for dynamic rollout budget allocation.
"""

import heapq
import json
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict


class BaseBudgetAllocator:
    """Base budget allocator with sliding window success rate tracking."""
    
    def __init__(
        self,
        n_low: int = 2,
        n_up: int = 32,
        smoothing_factor: float = 0.01,
        sliding_window_size: int = 3,
        **kwargs
    ):
        self.n_low = n_low
        self.n_up = n_up
        self.smoothing_factor = smoothing_factor
        self.sliding_window_size = sliding_window_size
        
        # Track per-sample statistics with sliding window
        self.task_stats: Dict[int, Dict] = defaultdict(lambda: {
            "successes": 0,
            "total": 0,
            "success_rate": None,
            "step_successes_list": [],
            "step_total_list": [],
            "current_step_successes": 0,
            "current_step_total": 0,
        })
        
        self.current_step = 0
    
    def allocate(self, global_indices: np.ndarray, total_budget: int) -> np.ndarray:
        """Allocate budget to samples."""
        raise NotImplementedError("Subclasses must implement allocate()")
    
    def update_stats(self, global_indices: np.ndarray, rewards: np.ndarray):
        """Update success rate statistics (reward=1.0 means success)."""
        for idx, reward in zip(global_indices, rewards):
            idx = int(idx)
            stats = self.task_stats[idx]
            stats["current_step_total"] += 1
            if reward == 1.0:
                stats["current_step_successes"] += 1
    
    def commit_step_stats(self):
        """Commit current step statistics to sliding window."""
        for idx, stats in self.task_stats.items():
            current_step_total = stats.get("current_step_total", 0)
            current_step_successes = stats.get("current_step_successes", 0)
            
            if current_step_total > 0:
                step_successes_list = stats.get("step_successes_list", [])
                step_total_list = stats.get("step_total_list", [])
                
                step_successes_list.append(current_step_successes)
                step_total_list.append(current_step_total)
                
                if len(step_successes_list) > self.sliding_window_size:
                    step_successes_list.pop(0)
                    step_total_list.pop(0)
                
                stats["step_successes_list"] = step_successes_list
                stats["step_total_list"] = step_total_list
                stats["successes"] = sum(step_successes_list)
                stats["total"] = sum(step_total_list)
                stats["success_rate"] = stats["successes"] / stats["total"] if stats["total"] > 0 else None
                stats["current_step_successes"] = 0
                stats["current_step_total"] = 0
        
        self.current_step += 1
    
    def get_success_rate(self, global_index: int) -> Optional[float]:
        """Get success rate for a sample."""
        stats = self.task_stats[global_index]
        return stats.get("success_rate")
    
    def save_stats(self, filepath: str):
        """Save statistics to file."""
        stats_dict = {}
        for idx, stats in self.task_stats.items():
            stats_dict[str(idx)] = {
                "successes": stats["successes"],
                "total": stats["total"],
                "success_rate": stats["success_rate"],
                "step_successes_list": stats.get("step_successes_list", []),
                "step_total_list": stats.get("step_total_list", []),
                "current_step_successes": stats.get("current_step_successes", 0),
                "current_step_total": stats.get("current_step_total", 0),
            }
        
        save_data = {
            "current_step": self.current_step,
            "task_stats": stats_dict
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Saved allocator stats to {filepath} (current_step={self.current_step})")
    
    def load_stats(self, filepath: str, resume: bool = False):
        """Load statistics from file.
        
        Args:
            resume: If True, restore full training state; if False, only load prior info.
        """
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        stats_dict = save_data.get("task_stats", {})
        self.task_stats.clear()
        
        for idx_str, stats in stats_dict.items():
            idx = int(idx_str)
            
            if resume:
                step_successes_list = stats.get("step_successes_list", [])
                step_total_list = stats.get("step_total_list", [])
                
                if len(step_successes_list) > self.sliding_window_size:
                    step_successes_list = step_successes_list[-self.sliding_window_size:]
                    step_total_list = step_total_list[-self.sliding_window_size:]
                
                window_successes = sum(step_successes_list)
                window_total = sum(step_total_list)
                
                self.task_stats[idx] = {
                    "successes": window_successes,
                    "total": window_total,
                    "success_rate": window_successes / window_total if window_total > 0 else None,
                    "step_successes_list": step_successes_list,
                    "step_total_list": step_total_list,
                    "current_step_successes": stats.get("current_step_successes", 0),
                    "current_step_total": stats.get("current_step_total", 0),
                }
            else:
                successes = stats.get("successes", 0)
                total = stats.get("total", 0)
                step_successes_list = stats.get("step_successes_list", [])
                step_total_list = stats.get("step_total_list", [])
                
                if not step_successes_list and total > 0:
                    step_successes_list = [successes]
                    step_total_list = [total]
                
                if len(step_successes_list) > self.sliding_window_size:
                    step_successes_list = step_successes_list[-self.sliding_window_size:]
                    step_total_list = step_total_list[-self.sliding_window_size:]
                
                window_successes = sum(step_successes_list)
                window_total = sum(step_total_list)
                
                self.task_stats[idx] = {
                    "successes": window_successes,
                    "total": window_total,
                    "success_rate": window_successes / window_total if window_total > 0 else None,
                    "step_successes_list": step_successes_list,
                    "step_total_list": step_total_list,
                    "current_step_successes": 0,
                    "current_step_total": 0,
                }
        
        if resume:
            self.current_step = save_data.get("current_step", 0)
            print(f"Resumed allocator from {filepath} (current_step={self.current_step}, samples={len(self.task_stats)})")
        else:
            self.current_step = 0
            print(f"Loaded allocator stats from {filepath} (loaded {len(self.task_stats)} samples, training state reset)")


class BetaAllocator(BaseBudgetAllocator):
    """Beta distribution PDF-based budget allocator with heap greedy algorithm.
    
    Uses sample-based sliding window to update Beta parameters and supports
    exploration budget with fallback mechanism.
    """
    
    def __init__(
        self,
        n_low: int = 2,
        n_up: int = 32,
        sliding_window_size: int = 3,
        smoothing_factor: float = 0.01,
        tau_min: float = 3.0,
        tau_max: float = 12.0,
        var_max: float = 0.1667,
        exploration_budget_ratio: float = 0.8,
        remaining_budget_method: str = "weighted",
        weight_alpha: float = 1.0,
        beta_params_sum: float = 7,
        **kwargs
    ):
        super().__init__(n_low, n_up, smoothing_factor, sliding_window_size, **kwargs)
        
        self.beta_alpha_semantic = "exploit2explore"
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.var_max = var_max
        self.tau_global = None
        self.exploration_budget_ratio = exploration_budget_ratio
        self.remaining_budget_method = remaining_budget_method
        self.weight_alpha = weight_alpha
        self.beta_params_sum = beta_params_sum
        self.beta_params_scale = self.beta_params_sum - 2
        
        # Beta parameters (exploit2explore: alpha corresponds to failure rate)
        self.beta_alpha = self.beta_params_sum - 2.0
        self.beta_beta = 2.0
        
        self.sample_based_step_stats_window = []
        self.current_global_successes = 0
        self.current_global_failures = 0
        self.current_step_sample_success_rates = []
    
    def update_stats(self, global_indices: np.ndarray, rewards: np.ndarray):
        """Update statistics and collect global success/failure counts."""
        super().update_stats(global_indices, rewards)
        
        for reward in rewards:
            if reward == 1.0:
                self.current_global_successes += 1
            else:
                self.current_global_failures += 1
    
    def _collect_sample_success_rates_from_task_stats(self):
        """Collect per-sample success rates from current step."""
        for idx, stats in self.task_stats.items():
            current_step_total = stats.get("current_step_total", 0)
            current_step_successes = stats.get("current_step_successes", 0)
            
            if current_step_total > 0:
                sample_success_rate = current_step_successes / current_step_total
                self.current_step_sample_success_rates.append(sample_success_rate)
    
    def _apply_adaptive_nonlinear_transform(self, success_rate: float, failure_rate: float):
        """Apply sigmoid transform when failure_rate <= 0.5 (late training stage).
        
        Formula: 1/(1+exp(-10*(x-0.5)))
        """
        transformed_success_rate = success_rate
        transformed_failure_rate = failure_rate
        
        if failure_rate <= 0.5:
            transformed_failure_rate = 1.0 / (1.0 + np.exp(-10.0*(failure_rate - 0.5)))
        
        return transformed_success_rate, transformed_failure_rate
    
    def commit_step_stats(self):
        """Commit step stats and update Beta parameters."""
        self._collect_sample_success_rates_from_task_stats()
        self._update_beta_params_sample_based_sliding_window()
        
        self.current_global_successes = 0
        self.current_global_failures = 0
        self.current_step_sample_success_rates.clear()
        
        super().commit_step_stats()
    
    def _update_beta_params_sample_based_sliding_window(self):
        """Update Beta parameters using sample-based sliding window."""
        if len(self.current_step_sample_success_rates) == 0:
            return
        
        sample_success_rates = np.array(self.current_step_sample_success_rates)
        avg_success_rate = np.mean(sample_success_rates)
        avg_failure_rate = 1.0 - avg_success_rate
        num_samples = len(sample_success_rates)
        
        self.sample_based_step_stats_window.append({
            "avg_success_rate": avg_success_rate,
            "avg_failure_rate": avg_failure_rate,
            "num_samples": num_samples
        })
        
        if len(self.sample_based_step_stats_window) > self.sliding_window_size:
            self.sample_based_step_stats_window.pop(0)
        
        window_avg_success_rate = np.mean([step["avg_success_rate"] for step in self.sample_based_step_stats_window])
        window_avg_failure_rate = np.mean([step["avg_failure_rate"] for step in self.sample_based_step_stats_window])
        
        transformed_success_rate, transformed_failure_rate = self._apply_adaptive_nonlinear_transform(
            window_avg_success_rate, window_avg_failure_rate
        )
        
        if transformed_failure_rate != window_avg_failure_rate:
            print(f"[BetaAllocator] Applied nonlinear transform (exploit2explore): "
                  f"{window_avg_failure_rate:.3f} -> {transformed_failure_rate:.3f}")
        
        alpha_min = 1.0
        alpha_max = self.beta_params_sum - 1.0
        
        # Alpha corresponds to failure rate in exploit2explore semantic
        alpha = alpha_min + self.beta_params_scale * transformed_failure_rate
        alpha = max(alpha_min, min(alpha_max, alpha))
        beta = self.beta_params_sum - alpha
        
        self.beta_alpha = alpha
        self.beta_beta = beta
        
        transform_info = f"->transformed:{transformed_failure_rate:.3f}" if transformed_failure_rate != window_avg_failure_rate else ""
        print(f"[BetaAllocator] Updated Beta params (exploit2explore): "
              f"alpha={alpha:.3f} (window_avg_failure_rate={window_avg_failure_rate:.3f}{transform_info}), "
              f"beta={beta:.3f} (window_avg_success_rate={window_avg_success_rate:.3f}) | "
              f"Current step: avg_success_rate={avg_success_rate:.3f}, {num_samples} samples | "
              f"Window: {len(self.sample_based_step_stats_window)} steps")
    
    def allocate(self, global_indices: np.ndarray, total_budget: int) -> np.ndarray:
        """Allocate budget using Beta PDF-based heap greedy algorithm."""
        M = len(global_indices)
        if M == 0:
            return np.array([])
        
        success_rates = [self.get_success_rate(int(idx)) for idx in global_indices]
        if all([i_succ is None for i_succ in success_rates]):
            base_alloc = total_budget // M
            remainder = total_budget % M
            allocation = np.full(M, base_alloc, dtype=int)
            allocation[:remainder] += 1
            return allocation

        success_rates = [i_succ if i_succ is not None else 0.5 for i_succ in success_rates]
        success_rates = np.array(success_rates)
        
        min_required = M * self.n_low
        if total_budget < min_required:
            base_alloc = total_budget // M
            remainder = total_budget % M
            allocation = np.full(M, base_alloc, dtype=int)
            allocation[:remainder] += 1
            return allocation
        
        # Step 1: Base allocation
        base_allocation = np.full(M, self.n_low, dtype=int)
        remaining_budget = total_budget - min_required
        
        if remaining_budget == 0:
            return base_allocation
        
        # Step 2: Categorize tasks
        very_hard_indices = []  # p < 0.05
        exploration_indices = []  # 0.05 <= p <= 0.99
        very_easy_indices = []  # p > 0.99
        exploration_rates = []
        
        for i, sr in enumerate(success_rates):
            if sr < 0.05:
                very_hard_indices.append(i)
            elif sr > 0.99:
                very_easy_indices.append(i)
            else:
                exploration_indices.append(i)
                exploration_rates.append(sr)
        
        # Step 3: Compute exploration budget
        if len(exploration_indices) > 0 and M > 0:
            if len(exploration_indices) > M / 2:
                budget_ratio = self.exploration_budget_ratio
            else:
                exploration_ratio = len(exploration_indices) / M
                min_ratio = 0.6
                max_ratio = self.exploration_budget_ratio
                budget_ratio = min_ratio + (max_ratio - min_ratio) * (exploration_ratio * 2)
        else:
            budget_ratio = 0.6
        
        exploration_budget = int(remaining_budget * budget_ratio)
        extra_allocation = np.zeros(M, dtype=int)
        
        # Step 4: Allocate exploration budget using heap greedy
        if len(exploration_indices) > 0 and exploration_budget > 0:
            exploration_rates_array = np.array(exploration_rates)
            exploration_allocation = self._allocate_beta_greedy_internal(
                exploration_rates_array, 
                len(exploration_indices) * self.n_low + exploration_budget
            )
            
            for i, idx in enumerate(exploration_indices):
                extra_allocation[idx] = exploration_allocation[i] - self.n_low
            
            allocated_exploration = np.sum(exploration_allocation) - len(exploration_indices) * self.n_low
            remaining_budget -= allocated_exploration
        else:
            allocated_exploration = 0
        
        # Step 5: Allocate remaining budget (fallback)
        if remaining_budget > 0:
            if len(very_hard_indices) > 0:
                remaining_budget = self._allocate_remaining_budget(
                    remaining_budget, very_hard_indices, success_rates,
                    extra_allocation, base_allocation
                )
            
            if remaining_budget > 0 and len(exploration_indices) > 0:
                remaining_budget = self._allocate_remaining_budget(
                    remaining_budget, exploration_indices, success_rates,
                    extra_allocation, base_allocation
                )
            
            if remaining_budget > 0 and len(very_easy_indices) > 0:
                remaining_budget = self._allocate_remaining_budget(
                    remaining_budget, very_easy_indices, success_rates,
                    extra_allocation, base_allocation
                )
        
        final_allocation = base_allocation + extra_allocation
        return final_allocation
    
    def _update_tau_global(self, success_rates: np.ndarray) -> float:
        """Dynamically compute tau based on success rate variance."""
        p_arr = np.clip(success_rates, self.smoothing_factor, 1.0 - self.smoothing_factor)
        mean_p = np.mean(p_arr)
        var_p = np.mean((p_arr - mean_p) ** 2)
        
        v = var_p / self.var_max
        v = float(np.clip(v, 0.0, 1.0))
        
        tau = self.tau_min + (self.tau_max - self.tau_min) * v
        self.tau_global = tau
        
        print(f"[BetaAllocator] Updated tau: mean_p={mean_p:.4f}, var_p={var_p:.6f}, "
              f"normalized_v={v:.4f}, tau={tau:.4f} (tau_min={self.tau_min}, tau_max={self.tau_max})")
        
        return tau
    
    def _beta_function(self, alpha: float, beta: float) -> float:
        """Compute Beta function B(α, β) using log-gamma to avoid overflow."""
        from math import lgamma, exp
        log_beta = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
        return exp(log_beta)
    
    def _beta_pdf(self, p: float, alpha: float, beta: float) -> float:
        """Compute Beta distribution PDF: f(p; α, β) = [p^(α-1) * (1-p)^(β-1)] / B(α, β)"""
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        
        from math import log, exp
        
        try:
            log_pdf = (alpha - 1) * log(p) + (beta - 1) * log(1 - p)
            log_beta = log(self._beta_function(alpha, beta))
            pdf = exp(log_pdf - log_beta)
            
            if not np.isfinite(pdf):
                return 1e-10
            
            return pdf
        except Exception:
            return 1e-10
    
    def _compute_value(self, p: float, N: int, tau: float) -> float:
        """Compute value function: V(N, p) = (1 - exp(-N/tau * p * (1-p))) * Beta_PDF(p; α, β)"""
        p_smoothed = np.clip(p, self.smoothing_factor, 1.0 - self.smoothing_factor)
        
        info_gain = self._beta_pdf(p_smoothed, self.beta_alpha, self.beta_beta)
        uncertainty_reduction = 1 - np.exp(-N / tau * p_smoothed * (1 - p_smoothed))
        
        return uncertainty_reduction * info_gain
    
    def _allocate_remaining_budget(
        self,
        remaining_budget: int,
        task_indices: List[int],
        success_rates: np.ndarray,
        extra_allocation: np.ndarray,
        base_allocation: np.ndarray
    ) -> int:
        """Allocate remaining budget to specified tasks."""
        if remaining_budget <= 0 or len(task_indices) == 0:
            return remaining_budget
        
        if self.remaining_budget_method == "weighted":
            return self._allocate_remaining_weighted(
                remaining_budget, task_indices, success_rates, extra_allocation, base_allocation
            )
        else:
            return self._allocate_remaining_uniform(
                remaining_budget, task_indices, extra_allocation, base_allocation
            )
    
    def _allocate_remaining_uniform(
        self,
        remaining_budget: int,
        task_indices: List[int],
        extra_allocation: np.ndarray,
        base_allocation: np.ndarray
    ) -> int:
        """Uniform allocation of remaining budget."""
        if remaining_budget <= 0 or len(task_indices) == 0:
            return remaining_budget
        
        available_indices = []
        for idx in task_indices:
            current_total = base_allocation[idx] + extra_allocation[idx]
            if current_total < self.n_up:
                available_indices.append(idx)
        
        if len(available_indices) == 0:
            return remaining_budget
        
        budget_per_task = remaining_budget // len(available_indices)
        remainder = remaining_budget % len(available_indices)
        
        for i, idx in enumerate(available_indices):
            current_total = base_allocation[idx] + extra_allocation[idx]
            max_extra = self.n_up - current_total
            to_allocate = min(budget_per_task + (1 if i < remainder else 0), max_extra)
            extra_allocation[idx] += to_allocate
            remaining_budget -= to_allocate
        
        return remaining_budget
    
    def _allocate_remaining_weighted(
        self,
        remaining_budget: int,
        task_indices: List[int],
        success_rates: np.ndarray,
        extra_allocation: np.ndarray,
        base_allocation: np.ndarray
    ) -> int:
        """Weighted allocation of remaining budget based on (1-p)^weight_alpha."""
        if remaining_budget <= 0 or len(task_indices) == 0:
            return remaining_budget
        
        weights = []
        for idx in task_indices:
            current_total = base_allocation[idx] + extra_allocation[idx]
            if current_total < self.n_up:
                p = success_rates[idx]
                p_smoothed = np.clip(p, 1e-6, 1.0 - 1e-6)
                w = (1.0 - p_smoothed) ** self.weight_alpha
                weights.append((idx, w))
        
        if len(weights) == 0:
            return remaining_budget
        
        total_weight = sum(w for _, w in weights)
        if total_weight == 0:
            return self._allocate_remaining_uniform(
                remaining_budget, task_indices, extra_allocation, base_allocation
            )
        
        for idx, w in weights:
            theoretical_alloc = int(remaining_budget * w / total_weight)
            current_total = base_allocation[idx] + extra_allocation[idx]
            max_extra = self.n_up - current_total
            actual_alloc = min(theoretical_alloc, max_extra)
            extra_allocation[idx] += actual_alloc
            remaining_budget -= actual_alloc
        
        if remaining_budget > 0:
            available_indices = [idx for idx, _ in weights 
                                if base_allocation[idx] + extra_allocation[idx] < self.n_up]
            if len(available_indices) > 0:
                budget_per_task = remaining_budget // len(available_indices)
                remainder = remaining_budget % len(available_indices)
                
                for i, idx in enumerate(available_indices):
                    current_total = base_allocation[idx] + extra_allocation[idx]
                    max_extra = self.n_up - current_total
                    to_allocate = min(budget_per_task + (1 if i < remainder else 0), max_extra)
                    extra_allocation[idx] += to_allocate
                    remaining_budget -= to_allocate
        
        return remaining_budget
    
    def _allocate_beta_greedy_internal(self, success_rates: np.ndarray, total_budget: int) -> np.ndarray:
        """Beta heap greedy algorithm implementation."""
        K = len(success_rates)
        tau = self._update_tau_global(success_rates)
        
        N = np.full(K, self.n_low, dtype=int)
        current_total = np.sum(N)
        
        if current_total > total_budget:
            base_alloc = total_budget // K
            remainder = total_budget % K
            allocation = np.full(K, base_alloc, dtype=int)
            allocation[:remainder] += 1
            return allocation
        
        R = total_budget - current_total
        
        if R == 0:
            return N
        
        # Build max heap
        heap = []
        for i in range(K):
            if N[i] < self.n_up:
                current_value = self._compute_value(success_rates[i], N[i], tau)
                next_value = self._compute_value(success_rates[i], N[i] + 1, tau)
                marginal_gain = next_value - current_value
                heapq.heappush(heap, (-marginal_gain, i))
        
        # Iterative allocation
        while R > 0 and heap:
            neg_gain, i = heapq.heappop(heap)
            N[i] += 1
            R -= 1
            
            if N[i] < self.n_up:
                current_value = self._compute_value(success_rates[i], N[i], tau)
                next_value = self._compute_value(success_rates[i], N[i] + 1, tau)
                new_marginal_gain = next_value - current_value
                heapq.heappush(heap, (-new_marginal_gain, i))
        
        return N
