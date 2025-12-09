"""
VAE 驱动的遗传算法引擎。

核心想法：在潜变量空间进行变异与交叉，避免仅对单个音符做随机修改。
需要一个基于 VAE 的评估器实例 (VAE.vae_evaluator.MusicEvaluator)，以便访问模型进行编码/解码。
"""

import numpy as np
import random
import torch

from MusicRep import MelodySequence, MusicConfig, fixGrid


class MusicGeneticOptimizerVAE:
    def __init__(
        self,
        vae_evaluator,
        pop_size=100,
        n_generations=80,
        elite_ratio=0.1,
        # 传统结构性算子概率
        prob_transposition=0.05,
        prob_retrograde=0.03,
        prob_inversion=0.03,
        prob_point_mutation=0.05,
        # 潜空间相关超参
        prob_latent_mutation=0.7,
        latent_sigma=0.8,
        pull_to_centroid=0.25,
        prob_latent_crossover=1.0,
        device="cpu",
    ):
        self.evaluator = vae_evaluator  # 需提供 .model, .target_centroid (可选)
        self.device = device if device else "cpu"

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.elite_size = max(1, int(pop_size * elite_ratio))

        self.prob_transposition = prob_transposition
        self.prob_retrograde = prob_retrograde
        self.prob_inversion = prob_inversion
        self.prob_point_mutation = prob_point_mutation

        self.prob_latent_mutation = prob_latent_mutation
        self.latent_sigma = latent_sigma
        self.pull_to_centroid = pull_to_centroid
        self.prob_latent_crossover = prob_latent_crossover

        self.population = []
        self.history_ = {"max_fitness": [], "avg_fitness": []}
        self.best_individual_ = None
        self.best_fitness_ = None

    # ----------------------- 初始化与修复 -----------------------
    def _initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = fixGrid(MelodySequence.from_random().grid)
            self.population.append(ind)
        self.population = np.array(self.population)

    def _repair(self, gene):
        return fixGrid(gene)

    # ----------------------- VAE 编码/解码 -----------------------
    def _encode(self, gene: np.ndarray):
        tensor = torch.as_tensor(gene, dtype=torch.long, device=self.evaluator.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.evaluator.model.encode(tensor)
        return mu.squeeze(0)

    def _decode(self, z: torch.Tensor):
        z = z.unsqueeze(0)
        with torch.no_grad():
            decoded = self.evaluator.model.decode(z)  # 自回归解码
        return decoded.squeeze(0).cpu().numpy()

    # ----------------------- 结构性算子 -----------------------
    def _op_transposition(self, gene):
        new_gene = gene.copy()
        shift = np.random.randint(-5, 6)
        if shift == 0:
            return new_gene
        mask = new_gene > MusicConfig.HOLD_VAL
        if np.any(mask):
            shifted = new_gene[mask] + shift
            if np.min(shifted) >= MusicConfig.PITCH_MIN and np.max(shifted) <= MusicConfig.PITCH_MAX:
                new_gene[mask] = shifted
        return new_gene

    def _op_retrograde(self, gene):
        return self._repair(gene[::-1])

    def _op_inversion(self, gene):
        new_gene = gene.copy()
        valid_idx = np.where(new_gene > MusicConfig.HOLD_VAL)[0]
        if len(valid_idx) == 0:
            return new_gene
        pivot = new_gene[valid_idx[0]]
        mask = new_gene > MusicConfig.HOLD_VAL
        inverted_vals = 2 * pivot - new_gene[mask]
        if np.all((inverted_vals >= MusicConfig.PITCH_MIN) & (inverted_vals <= MusicConfig.PITCH_MAX)):
            new_gene[mask] = inverted_vals
        return new_gene

    def _op_point_mutation(self, gene):
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

    # ----------------------- 潜空间算子 -----------------------
    def _latent_mutation(self, gene):
        z = self._encode(gene)
        noise = torch.randn_like(z) * self.latent_sigma
        z_mut = z + noise

        # 若 evaluator 预先设置了 target_centroid，则可将 latent 拉向目标风格
        if getattr(self.evaluator, "target_centroid", None) is not None:
            z_mut = (1 - self.pull_to_centroid) * z_mut + self.pull_to_centroid * self.evaluator.target_centroid

        decoded = self._decode(z_mut)
        return self._repair(decoded)

    def _latent_crossover(self, parent1, parent2):
        z1 = self._encode(parent1)
        z2 = self._encode(parent2)
        w = torch.rand(1, device=z1.device)
        z_child = w * z1 + (1 - w) * z2
        decoded = self._decode(z_child)
        return self._repair(decoded)

    # ----------------------- 综合变异 -----------------------
    def _mutate(self, gene):
        current = gene.copy()

        # 结构性小变换
        if random.random() < self.prob_transposition:
            current = self._op_transposition(current)
        if random.random() < self.prob_retrograde:
            current = self._op_retrograde(current)
        if random.random() < self.prob_inversion:
            current = self._op_inversion(current)
        current = self._op_point_mutation(current)

        # 潜空间大变异
        if random.random() < self.prob_latent_mutation:
            current = self._latent_mutation(current)

        return current

    def _crossover(self, parent1, parent2):
        if random.random() < self.prob_latent_crossover:
            return self._latent_crossover(parent1, parent2)
        # 回退到简单单点交叉
        point = np.random.randint(1, len(parent1) - 1)
        child = np.concatenate([parent1[:point], parent2[point:]])
        return self._repair(child)

    # ----------------------- 适应度计算 -----------------------
    def _calculate_fitness_batch(self):
        return self.evaluator.evaluate(self.population)

    # ----------------------- 主流程 -----------------------
    def fit(self, verbose=True):
        self._initialize_population()

        for gen in range(self.n_generations):
            scores = self._calculate_fitness_batch()

            best_idx = np.argmax(scores)
            self.best_individual_ = self.population[best_idx].copy()
            self.best_fitness_ = scores[best_idx]
            self.history_["max_fitness"].append(scores[best_idx])
            self.history_["avg_fitness"].append(np.mean(scores))

            if verbose:
                print(f"Gen {gen+1}: Best={scores[best_idx]:.4f}, Avg={np.mean(scores):.4f}")

            # 生成下一代
            next_pop = []
            elite_indices = np.argsort(scores)[-self.elite_size:]
            for idx in elite_indices:
                next_pop.append(self.population[idx].copy())

            while len(next_pop) < self.pop_size:
                idx1 = self._tournament_select(scores)
                idx2 = self._tournament_select(scores)

                child = self._crossover(self.population[idx1], self.population[idx2])
                child = self._mutate(child)
                next_pop.append(child)

            self.population = np.array(next_pop[: self.pop_size])

        return self

    def _tournament_select(self, scores, k=3):
        candidates = np.random.randint(0, self.pop_size, k)
        return candidates[np.argmax(scores[candidates])]

    def predict(self):
        if self.best_individual_ is None:
            raise ValueError("Not fitted")
        return MelodySequence(self.best_individual_)