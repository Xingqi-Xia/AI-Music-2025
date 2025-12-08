import numpy as np
import copy
import random
# 确保 fix_grid 已经从 MusicRep 导入 (如果没导入，请把之前的 repair 逻辑放回来)
from MusicRep import MelodySequence, MusicConfig, fixGrid

class MusicGeneticOptimizer:
    """
    一个模仿 Sklearn 风格的遗传算法优化器。
    集成了高级音乐变换算子：移调、逆行、倒影。
    """
    def __init__(self, 
                 pop_size=100, 
                 n_generations=50, 
                 elite_ratio=0.05,
                 # --- 概率参数配置 ---
                 prob_point_mutation=0.1,  # 基础点变异概率
                 prob_transposition=0.1,   # 移调概率
                 prob_retrograde=0.05,     # 逆行概率
                 prob_inversion=0.05,      # 倒影概率
                 # ------------------
                 evaluator_model=None,
                 device='cpu'):
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.elite_size = max(1, int(pop_size * elite_ratio))
        
        # 保存各种算子的概率
        self.prob_point_mutation = prob_point_mutation
        self.prob_transposition = prob_transposition
        self.prob_retrograde = prob_retrograde
        self.prob_inversion = prob_inversion
        
        self.evaluator = evaluator_model
        if self.evaluator is None:
            print("Warning: No evaluator model provided, using default rules.")
        self.device = device
        
        self.population = []
        self.history_ = {'max_fitness': [], 'avg_fitness': []}
        self.best_individual_ = None

    def _initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            # 初始化后立即修复，保证起点合法
            ind = fixGrid(MelodySequence.from_random().grid)
            self.population.append(ind)
        self.population = np.array(self.population)

    def _repair(self, gene):
        """调用外部库的修复逻辑"""
        return fixGrid(gene)

    def _calculate_fitness_batch(self):
        """(保持不变) 批量计算适应度"""
        if self.evaluator is None:
            # Fallback CPU 规则
            scores = []
            for gene in self.population:
                # 简单规则示例：奖励有效音符数 + 调性
                active = np.sum(gene > 1)
                if active == 0:
                    scores.append(0.0)
                    continue
                score = active / 32.0
                # C Major 奖励
                c_scale = {0, 2, 4, 5, 7, 9, 11}
                in_key = sum(1 for x in gene if x > 1 and (x % 12) in c_scale)
                score += (in_key / active) * 0.5
                scores.append(score)
            return np.array(scores)
        else:
            # 这里接入你之前的 Evaluator 类逻辑
            return self.evaluator.evaluate(self.population)
            pass

    # ==========================================
    # 高级音乐算子 (Advanced Musical Operators)
    # ==========================================

    def _op_transposition(self, gene):
        """【移调】整个序列升高或降低 k 个半音"""
        new_gene = gene.copy()
        shift = np.random.randint(-5, 6) # -5 到 +5 半音
        if shift == 0: return new_gene
        
        # 只对音符(>1)操作，不影响 Rest(0) 和 Hold(1)
        mask = new_gene > MusicConfig.HOLD_VAL
        
        if np.any(mask):
            shifted = new_gene[mask] + shift
            # 边界检查：必须所有音都在合法范围内才执行移调
            if np.min(shifted) >= MusicConfig.PITCH_MIN and \
               np.max(shifted) <= MusicConfig.PITCH_MAX:
                new_gene[mask] = shifted
                
        return new_gene

    def _op_retrograde(self, gene):
        """【逆行】时间轴翻转"""
        # 直接翻转数组
        new_gene = gene[::-1]
        # 注意：逆行可能会导致 Hold 出现在 Rest 后面或开头，必须 Repair
        return self._repair(new_gene)

    def _op_inversion(self, gene):
        """【倒影/影子】以某个音为轴，进行镜像翻转"""
        new_gene = gene.copy()
        
        # 1. 寻找轴心 (Pivot)
        # 通常取旋律的第一个音，或者平均音高。这里取第一个非休止音符。
        valid_indices = np.where(new_gene > MusicConfig.HOLD_VAL)[0]
        if len(valid_indices) == 0:
            return new_gene
            
        pivot_pitch = new_gene[valid_indices[0]]
        
        # 2. 计算倒影: New = Pivot + (Pivot - Old) = 2*Pivot - Old
        # 只有音高才参与倒影
        mask = new_gene > MusicConfig.HOLD_VAL
        inverted_vals = 2 * pivot_pitch - new_gene[mask]
        
        # 3. 边界检查与应用
        # 如果倒影后超出钢琴范围，这就不是一个合法的倒影，放弃操作
        if np.all((inverted_vals >= MusicConfig.PITCH_MIN) & 
                  (inverted_vals <= MusicConfig.PITCH_MAX)):
            new_gene[mask] = inverted_vals
            
        return new_gene

    def _op_point_mutation(self, gene):
        """【点变异】随机修改某些位置"""
        new_gene = gene.copy()
        for i in range(len(new_gene)):
            if np.random.random() < self.prob_point_mutation:
                r = np.random.random()
                if r < 0.2: 
                    new_gene[i] = MusicConfig.REST_VAL
                elif r < 0.4: 
                    new_gene[i] = MusicConfig.HOLD_VAL
                else: 
                    new_gene[i] = np.random.randint(MusicConfig.PITCH_MIN, MusicConfig.PITCH_MAX + 1)
        return self._repair(new_gene)

    # ==========================================
    # 核心流程
    # ==========================================

    def _mutate(self, gene):
        """
        综合变异调度器：
        根据概率决定应用哪些算子。可以同时应用多个。
        """
        current_gene = gene.copy()
        
        # 1. 结构性变换 (Structural Changes)
        # 这些变换改变旋律的整体形态
        
        if np.random.random() < self.prob_transposition:
            current_gene = self._op_transposition(current_gene)
            
        if np.random.random() < self.prob_retrograde:
            current_gene = self._op_retrograde(current_gene)
            
        if np.random.random() < self.prob_inversion:
            current_gene = self._op_inversion(current_gene)
            
        # 2. 细节变换 (Detail Changes)
        # 总是尝试进行点变异（内部有概率控制每个音）
        current_gene = self._op_point_mutation(current_gene)
        
        return self._repair(current_gene)

    def _crossover(self, parent1, parent2):
        """单点交叉"""
        point = np.random.randint(1, len(parent1) - 1)
        c1 = np.concatenate([parent1[:point], parent2[point:]])
        c2 = np.concatenate([parent2[:point], parent1[point:]])
        return self._repair(c1), self._repair(c2)

    def fit(self, verbose=True):
        self._initialize_population()
        
        for generation in range(self.n_generations):
            # 1. 评估
            scores = self._calculate_fitness_batch()
            
            # 记录数据
            best_idx = np.argmax(scores)
            self.best_individual_ = self.population[best_idx].copy()
            self.best_fitness_ = scores[best_idx]
            self.history_['max_fitness'].append(scores[best_idx])
            self.history_['avg_fitness'].append(np.mean(scores))
            
            if verbose:
                print(f"Gen {generation+1}: Best={scores[best_idx]:.4f}")
                
            # 2. 生成下一代
            next_pop = []
            
            # 精英保留
            elite_indices = np.argsort(scores)[-self.elite_size:]
            for idx in elite_indices:
                next_pop.append(self.population[idx].copy())
                
            # 繁衍
            while len(next_pop) < self.pop_size:
                # 锦标赛选择
                idx1 = self._tournament_select(scores)
                idx2 = self._tournament_select(scores)
                
                # 交叉
                c1, c2 = self._crossover(self.population[idx1], self.population[idx2])
                
                # 变异 (包含移调、倒影等逻辑)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                next_pop.extend([c1, c2])
                
            self.population = np.array(next_pop[:self.pop_size])
            
        return self

    def _tournament_select(self, scores, k=3):
        candidates = np.random.randint(0, self.pop_size, k)
        best_cand = candidates[np.argmax(scores[candidates])]
        return best_cand

    def predict(self):
        if self.best_individual_ is None:
            raise ValueError("Not fitted")
        return MelodySequence(self.best_individual_)