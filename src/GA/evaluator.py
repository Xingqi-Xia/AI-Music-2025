import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Union

# 尝试导入 PyTorch，如果没有安装也不影响规则评估器的使用
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==========================================
# 1. 抽象基类 (Interface)
# ==========================================
class MusicEvaluator(ABC):
    """
    所有评估器的基类。
    """
    @abstractmethod
    def evaluate(self, population_grid: np.ndarray) -> np.ndarray:
        """
        核心接口。
        
        Args:
            population_grid: 形状为 (pop_size, seq_len) 的 Numpy 整数数组。
            
        Returns:
            scores: 形状为 (pop_size,) 的 Numpy 浮点数组，表示适应度分数。
        """
        pass


class RuleBasedEvaluator(MusicEvaluator):
    """
    使用一组加权的 Python 函数来评估旋律序列的评估器。
    这玩意是随便瞎写的，主要是测试遗传算法流程，实际我们要好好写它
    """
    def __init__(self):
        self.rules: List[Callable] = []
        self.weights: List[float] = []

    def add_rule(self, rule_func: Callable[[np.ndarray], float], weight: float = 1.0):
        """
        注册一个新规则。
        rule_func: 接收单个 grid (shape: 32,)，返回 float 分数。
        """
        self.rules.append(rule_func)
        self.weights.append(weight)
        return self # 支持链式调用

    def evaluate(self, population_grid: np.ndarray) -> np.ndarray:
        pop_size = len(population_grid)
        scores = np.zeros(pop_size)

        # 遍历种群中的每个个体
        for i in range(pop_size):
            individual_grid = population_grid[i]
            total_score = 0.0
            
            # 遍历所有规则进行加权求和
            for rule, weight in zip(self.rules, self.weights):
                total_score += rule(individual_grid) * weight
            
            scores[i] = total_score
            
        return scores

# ==========================================
# 3. 预定义的通用规则库 (Common Rules)
# ==========================================
class BasicRules:
    """
    一些通用的乐理规则函数，可以直接添加到 RuleBasedEvaluator 中。
    """
    
    @staticmethod
    def pitch_in_key_c_major(grid: np.ndarray) -> float:
        """奖励 C 大调音阶内的音符"""
        # C Major: 0, 2, 4, 5, 7, 9, 11
        scale_set = {0, 2, 4, 5, 7, 9, 11}
        notes = grid[grid > 1] # 排除 Rest(0) 和 Hold(1)
        if len(notes) == 0: return 0.0
        
        count = sum(1 for n in notes if (n % 12) in scale_set)
        return count / len(notes)

    @staticmethod
    def rhythmic_variety(grid: np.ndarray) -> float:
        """奖励节奏变化，惩罚过多休止或全屏延音"""
        # 计算非 Hold 的事件数量 (即 Note On 或 Rest)
        changes = np.sum(grid != 1)
        # 我们希望变化率适中，比如占 20%~80%
        ratio = changes / len(grid)
        if 0.2 <= ratio <= 0.8:
            return 1.0
        else:
            return 0.2

    @staticmethod
    def smooth_contour(grid: np.ndarray) -> float:
        """奖励平滑的音程进行，惩罚大跳"""
        notes = grid[grid > 1]
        if len(notes) < 2: return 0.0
        
        jumps = 0
        for k in range(len(notes) - 1):
            diff = abs(notes[k] - notes[k+1])
            if diff > 7: # 大于纯五度
                jumps += 1
        
        # 跳跃越少分越高
        return max(0, 1.0 - (jumps * 0.1))

