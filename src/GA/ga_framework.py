# 一个通用的遗传算法框架，支持多规则评估器和变异调度器
# 和上一个版本比更加模块化，易于扩展和定制
import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Callable, Optional, Union

# ==========================================
# Part A: 基础组件 & 接口 (Base Components)
# ==========================================

class Individual(ABC):
    """个体基类"""
    def __init__(self, data: Any):
        self.data = data
        self.fitness: float = 0.0

    @abstractmethod
    def copy(self):
        pass

class Evaluator(ABC):
    """评估器接口, 应该返回一个与 population 长度相同的分数数组"""
    @abstractmethod
    def evaluate(self, population: List[Individual]) -> np.array:
        pass

class MultiRuleEvaluator(Evaluator):
    """
    【新增】内置的多规则评估器。
    支持注册多个打分函数（规则），并按权重求和。
    规则函数签名应为: func(data: Any) -> float
    """
    def __init__(self):
        # 存储格式: [(rule_func, weight, rule_name), ...]
        self.rules: List[Tuple[Callable, float, str]] = []

    def register(self, rule_func: Callable[[Any], float], weight: float = 1.0, name: str = None):
        """
        注册一条规则。
        rule_func: 接收 individual.data，返回 float 分数。
        weight: 权重，默认为 1.0。
        """
        rule_name = name if name else rule_func.__name__
        self.rules.append((rule_func, weight, rule_name))

    def get_rule_names(self) -> List[str]:
        return [rule_name for _, _, rule_name in self.rules]

    def evaluate_with_breakdown(self, population: List[Individual]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.rules:
            zeros = np.zeros(len(population))
            return zeros, np.zeros((0, len(population)))

        total_scores = np.zeros(len(population), dtype=float)
        breakdown = np.zeros((len(self.rules), len(population)), dtype=float)

        for j, ind in enumerate(population):
            for i, (func, weight, _) in enumerate(self.rules):
                raw_score = func(ind.data)
                weighted_score = raw_score * weight
                breakdown[i, j] = weighted_score
                total_scores[j] += weighted_score

        return total_scores, breakdown

    def evaluate(self, population: List[Individual]) -> np.array:
        total_scores, _ = self.evaluate_with_breakdown(population)
        return total_scores

class SelectionStrategy(ABC):
    """选择策略接口"""
    @abstractmethod
    def select(self, population: List[Individual], fitness_scores: np.array) -> Individual:
        pass

class CrossoverStrategy(ABC):
    """交叉策略接口"""
    @abstractmethod
    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        pass

class MutationStrategy(ABC):
    """变异策略接口"""
    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        pass

# ==========================================
# Part B: 调度器 (Scheduler)
# ==========================================

class MutationScheduler:
    """变异调度器：管理多个变异算子"""
    def __init__(self):
        self.strategies: List[MutationStrategy] = []
        self.weights: List[float] = []
        self.names: List[str] = []

    def register(self, strategy: MutationStrategy, weight: float, name: str = None):
        self.strategies.append(strategy)
        self.weights.append(weight)
        self.names.append(name if name else strategy.__class__.__name__)

    def mutate(self, individual: Individual) -> Individual:
        if not self.strategies:
            return individual
        # 轮盘赌选择
        chosen_strategy = random.choices(self.strategies, weights=self.weights, k=1)[0]
        return chosen_strategy.mutate(individual)

# ==========================================
# Part C: 核心引擎 (Core Engine)
# ==========================================

class GAEngine:
    """通用遗传算法引擎"""
    def __init__(self, 
                 pop_size: int,
                 n_generations: int,
                 evaluator: Evaluator,
                 selection_strat: SelectionStrategy,
                 crossover_strat: CrossoverStrategy,
                 mutation_scheduler: MutationScheduler,
                 individual_factory: Callable[[], Individual],
                 repair_func: Optional[Callable[[Any], Any]] = None,
                 elite_ratio: float = 0.05):
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.evaluator = evaluator
        self.selection_strat = selection_strat
        self.crossover_strat = crossover_strat
        self.mutation_scheduler = mutation_scheduler
        self.individual_factory = individual_factory
        self.repair_func = repair_func
        self.elite_size = max(1, int(pop_size * elite_ratio))
        
        self.population: List[Individual] = []
        self.history = {'best_fitness': [], 'avg_fitness': []}
        self.best_individual: Optional[Individual] = None
        self.best_individual_index: Optional[int] = None

    def _repair(self, ind: Individual) -> Individual:
        if self.repair_func:
            ind.data = self.repair_func(ind.data)
        return ind

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = self.individual_factory()
            self.population.append(self._repair(ind))
        print(f"🌱 Population initialized with {self.pop_size} individuals.")

    def step(self, generation_idx: int):
        # 1. 评估
        rule_breakdown = None
        rule_names = None

        if hasattr(self.evaluator, "evaluate_with_breakdown"):
            scores, rule_breakdown = self.evaluator.evaluate_with_breakdown(self.population)
            if hasattr(self.evaluator, "get_rule_names"):
                rule_names = self.evaluator.get_rule_names()
        else:
            scores = self.evaluator.evaluate(self.population)
        
        # 统计
        best_idx = np.argmax(scores)
        avg_score = np.mean(scores)
        current_best = self.population[best_idx]
        current_best.fitness = scores[best_idx]
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
            self.best_individual_index = best_idx
            self.best_individual.fitness = current_best.fitness
        
        self.history['best_fitness'].append(scores[best_idx])
        self.history['avg_fitness'].append(avg_score)
        
        print(f"Gen {generation_idx+1}/{self.n_generations} | Best: {scores[best_idx]:.4f} | Avg: {avg_score:.4f}")

        if rule_breakdown is not None and rule_names:
            best_rule_scores = rule_breakdown[:, best_idx]
            rule_parts = [f"{name}:{score:.4f}" for name, score in zip(rule_names, best_rule_scores)]
            print("      Rule breakdown -> " + " | ".join(rule_parts))

        # 2. 精英保留
        sorted_indices = np.argsort(scores)[::-1]
        next_pop = []
        for i in range(self.elite_size):
            elite = self.population[sorted_indices[i]].copy()
            next_pop.append(elite)

        # 3. 繁衍
        while len(next_pop) < self.pop_size:
            p1 = self.selection_strat.select(self.population, scores)
            p2 = self.selection_strat.select(self.population, scores)
            c1, c2 = self.crossover_strat.cross(p1, p2)
            
            c1 = self.mutation_scheduler.mutate(c1)
            c2 = self.mutation_scheduler.mutate(c2)
            
            next_pop.append(self._repair(c1))
            if len(next_pop) < self.pop_size:
                next_pop.append(self._repair(c2))
        
        self.population = next_pop

    def run(self):
        self.initialize()
        for i in range(self.n_generations):
            self.step(i)
        return self.best_individual

# ==========================================
# Part D: 规则基的具体实现 (Rule-Based Implementation)
# ==========================================

# 1. 音乐个体定义
class MusicIndividual(Individual):
    def copy(self):
        new_obj = MusicIndividual(self.data.copy())
        new_obj.fitness = self.fitness
        return new_obj



# 3. 锦标赛选择
class TournamentSelection(SelectionStrategy):
    def __init__(self, k=3):
        self.k = k

    def select(self, population: List[Individual], fitness_scores: np.array) -> Individual:
        indices = np.random.randint(0, len(population), self.k)
        best_idx = indices[np.argmax(fitness_scores[indices])]
        return population[best_idx] # 返回引用即可，交叉时会copy
