import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Union
from MusicRep import MusicConfig

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

class PentatonicRules:
    """
    五声调式
    """

    # =========================
    # Helper
    # =========================

    @staticmethod
    def _extract_attacks(grid: np.ndarray):
        g = np.asarray(grid, dtype=int)
        pitches = []
        steps = []
        i = 0
        while i < len(g):
            if g[i] > 1:
                pitches.append(int(g[i]))
                steps.append(i)
                i += 1
                while i < len(g) and g[i] == 1:
                    i += 1
            else:
                i += 1
        return steps, pitches

    @staticmethod
    def _pentatonic_set(tonic_pc: int):
        intervals = [0, 2, 4, 7, 9]
        return {(tonic_pc + x) % 12 for x in intervals}

    @staticmethod
    def _best_pentatonic_key(grid: np.ndarray):
        notes = [x for x in grid if x > 1]
        if not notes:
            return 0, 0.0

        pcs = [(n % 12) for n in notes]
        best_ratio = -1
        best_tonic = 0

        for tonic in range(12):
            scale = PentatonicRules._pentatonic_set(tonic)
            ratio = sum(pc in scale for pc in pcs) / len(pcs)
            if ratio > best_ratio:
                best_ratio = ratio
                best_tonic = tonic

        return best_tonic, float(best_ratio)

    # =========================
    # Rules
    # =========================

    @staticmethod
    def pentatonic_fit(grid: np.ndarray) -> float:
        """音是否落在五声调式中（最重要）"""
        _, ratio = PentatonicRules._best_pentatonic_key(grid)
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def stepwise_motion_preference(grid: np.ndarray) -> float:
        """级进偏好，小跳可接受，大跳扣分"""
        _, pitches = PentatonicRules._extract_attacks(grid)
        if len(pitches) < 2:
            return 0.0

        diffs = np.abs(np.diff(np.array(pitches)))
        score = 0.0
        for d in diffs:
            if d <= 2:
                score += 1.0
            elif d <= 5:
                score += 0.6
            elif d <= 9:
                score += 0.2
            else:
                score -= 0.5

        return float(np.clip(score / len(diffs), 0.0, 1.0))

    @staticmethod
    def melodic_flow(grid: np.ndarray) -> float:
        """旋律方向流畅性（不要上下疯狂反转）"""
        _, pitches = PentatonicRules._extract_attacks(grid)
        if len(pitches) < 4:
            return 0.0

        diffs = np.diff(np.array(pitches))
        signs = [1 if d > 0 else -1 for d in diffs if d != 0]
        if len(signs) < 3:
            return 0.5

        turns = sum(signs[i] != signs[i-1] for i in range(1, len(signs)))
        ratio = turns / len(signs)

        # 0.2~0.5 最自然
        return float(np.clip(1.0 - abs(ratio - 0.35) / 0.35, 0.0, 1.0))

    @staticmethod
    def rest_sparsity_penalty(grid: np.ndarray) -> float:
        """休止不要太多（五声怕碎）"""
        g = np.asarray(grid, dtype=int)
        rest_ratio = np.mean(g == 0)

        return float(1.0 - np.clip((rest_ratio - 0.15) / 0.30, 0.0, 1.0))

    @staticmethod
    def note_density_target(grid: np.ndarray) -> float:
        """起音数量：太密/太稀都不好"""
        steps, _ = PentatonicRules._extract_attacks(grid)
        n = len(steps)

        if 8 <= n <= 14:
            return 1.0
        if 6 <= n < 8:
            return 0.6 + 0.4 * (n - 6) / 2
        if 14 < n <= 18:
            return max(0.4, 1.0 - (n - 14) / 6)
        return 0.2

    @staticmethod
    def register_balance(grid: np.ndarray) -> float:
        """音区不要极端（五声太高/太低都不好听）"""
        _, pitches = PentatonicRules._extract_attacks(grid)
        if not pitches:
            return 0.0

        avg = np.mean(pitches)
        return float(np.clip(1.0 - abs(avg - 66) / 18.0, 0.4, 1.0))
    
    @staticmethod
    def overlong_note_penalty(grid: np.ndarray, max_dur: int = 4) -> float:
        """
        单个音持续太久会变单调
        max_dur: 允许的最大延音步数（默认 4 = 半音符）
        """
        g = np.asarray(grid, dtype=int)
        durs = []
        i = 0
        while i < len(g):
            if g[i] > 1:
                dur = 1
                i += 1
                while i < len(g) and g[i] == 1:
                    dur += 1
                    i += 1
                durs.append(dur)
            else:
                i += 1

        if not durs:
            return 0.0

        longest = max(durs)
        if longest <= max_dur:
            return 1.0

        # 超过就开始扣，超过很多会明显扣
        return float(np.clip(1.0 - (longest - max_dur) / 4.0, 0.0, 1.0))
    
    @staticmethod
    def contour_variation_reward(grid: np.ndarray) -> float:
        """鼓励出现适度的上下行变化（五声安全）"""
        _, pitches = PentatonicRules._extract_attacks(grid)
        if len(pitches) < 4:
            return 0.0

        diffs = np.diff(np.array(pitches))
        signs = []
        for d in diffs:
            if d > 0:
                signs.append(1)
            elif d < 0:
                signs.append(-1)

        if len(signs) < 3:
            return 0.5

        turns = sum(signs[i] != signs[i-1] for i in range(1, len(signs)))
        ratio = turns / len(signs)

        # 五声里：0.25~0.45 非常舒服
        return float(np.clip(1.0 - abs(ratio - 0.35) / 0.35, 0.0, 1.0))

class ClassicalRules:
    """
    古典调性
    """

    # --------------------------
    # Helper：提取起音序列/时值
    # --------------------------
    @staticmethod
    def _extract_attacks_and_durations(grid: np.ndarray):
        """
        返回：
          attacks_steps: List[int]    起音位置（grid[i] > HOLD）
          attacks_pitches: List[int]  起音音高（MIDI）
          attacks_durs: List[int]     该音持续步数（包含起音步）
        """
        g = np.asarray(grid, dtype=int)
        n = len(g)

        attacks_steps = []
        attacks_pitches = []
        attacks_durs = []

        i = 0
        while i < n:
            token = g[i]
            if token > MusicConfig.HOLD_VAL:  # 起音
                start = i
                pitch = int(token)
                dur = 1
                j = i + 1
                while j < n and g[j] == MusicConfig.HOLD_VAL:
                    dur += 1
                    j += 1
                attacks_steps.append(start)
                attacks_pitches.append(pitch)
                attacks_durs.append(dur)
                i = j
            else:
                i += 1

        return attacks_steps, attacks_pitches, attacks_durs

    # --------------------------
    # Helper：调性检测（best-of-keys）
    # --------------------------
    @staticmethod
    def _scale_pitch_classes(tonic_pc: int, mode: str):
        """
        返回该调式的音阶 pitch-class 集合
        mode: 'major' 或 'harmonic_minor'
        """
        tonic = tonic_pc % 12
        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]
        elif mode == "harmonic_minor":
            intervals = [0, 2, 3, 5, 7, 8, 11]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {(tonic + x) % 12 for x in intervals}

    @staticmethod
    def _best_key_major_or_harmonic_minor(grid: np.ndarray):
        """
        在 12 大调 + 12 和声小调里选“调内音比例”最高者
        返回：(best_tonic_pc, best_mode, best_ratio)
        """
        g = np.asarray(grid, dtype=int)
        notes = g[g > MusicConfig.HOLD_VAL]
        if notes.size == 0:
            return 0, "major", 0.0

        pcs = (notes % 12).astype(int)

        best_ratio = -1.0
        best_tonic = 0
        best_mode = "major"

        for tonic in range(12):
            for mode in ("major", "harmonic_minor"):
                scale_set = ClassicalRules._scale_pitch_classes(tonic, mode)
                in_key = np.sum(np.isin(pcs, list(scale_set)))
                ratio = float(in_key) / float(len(pcs))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_tonic = tonic
                    best_mode = mode

        return best_tonic, best_mode, float(best_ratio)

    @staticmethod
    def _degree_in_scale(pitch: int, tonic_pc: int, mode: str):
        """
        返回该音在该调式中的音级（1~7），若不在调内返回 None
        """
        pc = pitch % 12
        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]
        else:  # harmonic_minor
            intervals = [0, 2, 3, 5, 7, 8, 11]

        rel = (pc - (tonic_pc % 12)) % 12
        if rel not in intervals:
            return None
        return intervals.index(rel) + 1  # 1..7

    # =========================================================
    # Layer A
    # =========================================================

    @staticmethod
    def key_fit_best_of_24(grid: np.ndarray) -> float:
        """调内音比例（12大调 + 12和声小调取最高）"""
        _, _, ratio = ClassicalRules._best_key_major_or_harmonic_minor(grid)
        return float(ratio)

    @staticmethod
    def interval_stepwise_preference(grid: np.ndarray) -> float:
        """相邻起音音程偏好（2~5最优，>=10重扣）"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < 2:
            return 0.0

        diffs = np.abs(np.diff(np.array(pitches, dtype=int)))
        s = 0.0
        for d in diffs:
            if d <= 1:
                s += 0.7
            elif 2 <= d <= 5:
                s += 1.0
            elif 6 <= d <= 7:
                s += 0.6
            elif 8 <= d <= 9:
                s += 0.2
            else:  # >=10
                s -= 0.6

        s /= len(diffs)
        return float(np.clip(s, 0.0, 1.0))

    @staticmethod
    def rest_and_long_rest_penalty(grid: np.ndarray) -> float:
        """休止占比 + 最长连续休止惩罚（返回 0~1，1最好）"""
        g = np.asarray(grid, dtype=int)

        rest_ratio = float(np.mean(g == MusicConfig.REST_VAL))

        longest = 0
        cur = 0
        for x in g:
            if x == MusicConfig.REST_VAL:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0

        # rest_ratio：>0.45 变差；longest：>10 变差
        s1 = 1.0 - np.clip((rest_ratio - 0.20) / 0.25, 0.0, 1.0)
        s2 = 1.0 - np.clip((longest - 2) / 3.0, 0.0, 1.0)

        return float(np.clip(0.6 * s1 + 0.4 * s2, 0.0, 1.0))

    @staticmethod
    def melodic_range_score(grid: np.ndarray) -> float:
        """音域过窄扣大分（用起音范围）"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) == 0:
            return 0.0
        rng = int(np.max(pitches) - np.min(pitches))

        if rng < 7:
            return 0.0
        if rng <= 12:
            return float((rng - 7) / 5.0) * 0.7
        if rng <= 19:
            return 0.7 + float((rng - 12) / 7.0) * 0.3
        if rng <= 24:
            return 1.0
        return float(np.clip(1.0 - (rng - 24) / 12.0, 0.6, 1.0))
    
    @staticmethod
    def max_consecutive_rest_limit(grid: np.ndarray, max_len: int = 2) -> float:
        """连续休止上限：>max_len 快速扣分（0~1，1最好）"""
        g = np.asarray(grid, dtype=int)
        longest = 0
        cur = 0
        for x in g:
            if x == MusicConfig.REST_VAL:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        # 允许 2（1拍），超过开始急速掉
        return float(np.clip(1.0 - max(0, longest - max_len) / 3.0, 0.0, 1.0))


    # =========================================================
    # Layer B
    # =========================================================

    @staticmethod
    def phrase_start_on_strongbeat(grid: np.ndarray) -> float:
        """句首：step0 必须是起音（不能是0/1）"""
        g = np.asarray(grid, dtype=int)
        return 1.0 if g[0] > MusicConfig.HOLD_VAL else 0.0

    @staticmethod
    def cadence_end_stable_and_long(grid: np.ndarray) -> float:
        """句末：稳定音级（优先1/5/3）+ 靠近强拍 + 较长时值"""
        steps, pitches, durs = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) == 0:
            return 0.0

        tonic, mode, _ = ClassicalRules._best_key_major_or_harmonic_minor(grid)

        last_step = steps[-1]
        last_pitch = pitches[-1]
        last_dur = durs[-1]

        deg = ClassicalRules._degree_in_scale(last_pitch, tonic, mode)

        if deg == 1:
            deg_score = 1.0
        elif deg == 5:
            deg_score = 0.7
        elif deg == 3:
            deg_score = 0.5
        elif deg is None:
            deg_score = 0.0
        else:
            deg_score = 0.2

        beat_score = 1.0 if last_step in {24, 28} else 0.2  # 非强拍明显扣分

        if last_dur >= 4:
            dur_score = 1.0
        elif last_dur >= 2:
            dur_score = 0.7
        else:
            dur_score = 0.2

        return float(np.clip(0.45 * deg_score + 0.20 * beat_score + 0.35 * dur_score, 0.0, 1.0))

    @staticmethod
    def primary_degree_ratio(grid: np.ndarray) -> float:
        """主/下属/属（1/4/5级）占比目标约 0.55，偏离扣分"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) == 0:
            return 0.0

        tonic, mode, _ = ClassicalRules._best_key_major_or_harmonic_minor(grid)

        degrees = []
        for p in pitches:
            d = ClassicalRules._degree_in_scale(p, tonic, mode)
            if d is not None:
                degrees.append(d)

        if len(degrees) == 0:
            return 0.0

        ratio = float(np.mean([(d in (1, 4, 5)) for d in degrees]))
        score = 1.0 - np.clip(abs(ratio - 0.55) / 0.30, 0.0, 1.0)
        return float(score)

    @staticmethod
    def motif_repetition_interval(grid: np.ndarray, motif_len_attacks: int = 4) -> float:
        """动机重复：用开头若干起音的相对音程序列，在后续滑窗找最像的"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < motif_len_attacks + 2:
            return 0.0

        motif = pitches[:motif_len_attacks]
        motif_intervals = np.diff(np.array(motif, dtype=int))
        if motif_intervals.size == 0:
            return 0.0

        best = 0.0
        for start in range(1, len(pitches) - motif_len_attacks + 1):
            cand = pitches[start:start + motif_len_attacks]
            cand_intervals = np.diff(np.array(cand, dtype=int))
            if cand_intervals.size != motif_intervals.size:
                continue
            matches = np.mean(cand_intervals == motif_intervals)
            best = max(best, float(matches))

        return float(best)

    @staticmethod
    def syncopation_penalty(grid: np.ndarray) -> float:
        """切分惩罚：弱拍起音多扣分；弱拍起音跨强拍更扣分（返回 0~1，1最好）"""
        steps, _, durs = ClassicalRules._extract_attacks_and_durations(grid)
        if len(steps) == 0:
            return 0.0

        weak = 0
        cross_strong = 0

        for st, du in zip(steps, durs):
            if st % 2 == 1:
                weak += 1

                bar_start = (st // 8) * 8
                strong_in_bar = [bar_start + 0, bar_start + 4]
                cover_end = st + du
                for sp in strong_in_bar:
                    if st < sp < cover_end:
                        cross_strong += 1
                        break

        weak_ratio = weak / len(steps)
        cross_ratio = cross_strong / max(1, weak)

        penalty = np.clip(0.7 * weak_ratio + 0.3 * cross_ratio, 0.0, 1.0)
        return float(1.0 - penalty)

    @staticmethod
    def chromatic_semitone_overuse_penalty(grid: np.ndarray) -> float:
        """半音上下行过多扣分（abs(diff)==1 比例过高扣）"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < 2:
            return 0.0
        diffs = np.abs(np.diff(np.array(pitches, dtype=int)))
        ratio = float(np.mean(diffs == 1))

        score = 1.0 - np.clip((ratio - 0.15) / 0.20, 0.0, 1.0)
        return float(score)
    
    @staticmethod
    def bar_attack_density_smoothness(grid: np.ndarray) -> float:
        """相邻小节起音密度变化不要太大（0~1，1最好）"""
        steps, _, _ = ClassicalRules._extract_attacks_and_durations(grid)
        # 每小节 8 步
        counts = [0, 0, 0, 0]
        for st in steps:
            counts[st // 8] += 1
        diffs = [abs(counts[i] - counts[i-1]) for i in range(1, 4)]
        # 差 0~1 很好；差 2 还能接受；>=3 很割裂
        penalty = np.clip((np.mean(diffs) - 1.0) / 2.0, 0.0, 1.0)
        return float(1.0 - penalty)


    # =========================================================
    # Layer C（更像古典的细节：呼吸/高潮/解决）
    # =========================================================

    @staticmethod
    def note_density_target(grid: np.ndarray) -> float:
        """起音数量目标：太密或太稀都扣（古典旋律要有呼吸）"""
        steps, _, _ = ClassicalRules._extract_attacks_and_durations(grid)
        n = len(steps)
        # 32步里：10~16个起音比较像旋律；>18 往“机关枪”走
        if n < 6:
            return 0.0
        if 10 <= n <= 16:
            return 1.0
        if 7 <= n < 10:
            return 0.6 + 0.4 * (n - 7) / 3.0
        if 16 < n <= 20:
            return float(max(0.2, 1.0 - (n - 16) / 6.0))
        return 0.2

    @staticmethod
    def turning_points_target(grid: np.ndarray) -> float:
        """旋律转向次数（attack 音高方向变化）目标区间给高分"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < 4:
            return 0.0
        diffs = np.diff(np.array(pitches, dtype=int))
        # 忽略 0，防止同音造成噪声
        signs = []
        for d in diffs:
            if d > 0: signs.append(1)
            elif d < 0: signs.append(-1)
        if len(signs) < 3:
            return 0.0

        turns = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                turns += 1

        # 32步短句：6~12 个转向比较自然
        if 6 <= turns <= 12:
            return 1.0
        # 过少太平，过多太碎
        dist = min(abs(turns - 6), abs(turns - 12))
        return float(np.clip(1.0 - dist / 8.0, 0.2, 1.0))

    @staticmethod
    def climax_position_score(grid: np.ndarray) -> float:
        """最高音的出现位置接近 2/3（step≈21）加分"""
        steps, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) == 0:
            return 0.0
        max_pitch = max(pitches)
        # 取最高音最早出现的位置
        idx = pitches.index(max_pitch)
        pos = steps[idx]  # 0..31

        target = 21  # 2/3 * 32 ≈ 21
        # 距离越近越好：±2 步几乎满分，±8步以下还能接受
        dist = abs(pos - target)
        return float(np.clip(1.0 - dist / 10.0, 0.0, 1.0))

    @staticmethod
    def leading_tone_resolution(grid: np.ndarray) -> float:
        """导音(7级)→主音(1级)解决倾向加分（基于检测到的最佳调）"""
        _, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < 2:
            return 0.0

        tonic, mode, _ = ClassicalRules._best_key_major_or_harmonic_minor(grid)

        # 找到音级序列（不在调内的跳过）
        degs = []
        for p in pitches:
            d = ClassicalRules._degree_in_scale(p, tonic, mode)
            degs.append(d)

        lead_to_tonic = 0
        lead_total = 0
        for a, b in zip(degs[:-1], degs[1:]):
            if a == 7:
                lead_total += 1
                if b == 1:
                    lead_to_tonic += 1

        if lead_total == 0:
            return 0.5  # 没出现导音：中性
        return float(lead_to_tonic / lead_total)

    @staticmethod
    def dotted_rhythm_reward(grid: np.ndarray) -> float:
        """出现附点节奏（duration_steps=3）给小奖励"""
        _, _, durs = ClassicalRules._extract_attacks_and_durations(grid)
        if len(durs) == 0:
            return 0.0
        ratio = float(np.mean(np.array(durs) == 3))
        # 有一些就好，不需要很多
        return float(np.clip(ratio * 3.0, 0.0, 1.0))

    @staticmethod
    def pre_barline_rest_reward(grid: np.ndarray) -> float:
        """小节线前一拍（每小节 step6~7）出现休止，给小奖励（像换气）"""
        g = np.asarray(grid, dtype=int)
        reward = 0.0
        for bar_start in (0, 8, 16, 24):
            for s in (bar_start + 6, bar_start + 7):
                if 0 <= s < len(g) and g[s] == MusicConfig.REST_VAL:
                    reward += 1.0
                    break
        return float(np.clip(reward / 4.0, 0.0, 1.0))
    
    @staticmethod
    def contour_consistency_with_head(grid: np.ndarray) -> float:
        """
        轮廓一致性：以第1小节为参考，中段（第2-3小节）的上/下行比例不要反转
        """
        steps, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < 6:
            return 0.0

        # 把 attack 按小节分组
        by_bar = {0: [], 1: [], 2: [], 3: []}
        for st, p in zip(steps, pitches):
            by_bar[st // 8].append(p)

        def up_ratio(seq):
            if len(seq) < 2:
                return None
            diffs = np.diff(np.array(seq, dtype=int))
            ups = np.sum(diffs > 0)
            downs = np.sum(diffs < 0)
            total = ups + downs
            if total == 0:
                return 0.5
            return float(ups / total)

        head = up_ratio(by_bar[0])
        mid1 = up_ratio(by_bar[1])
        mid2 = up_ratio(by_bar[2])

        if head is None or mid1 is None or mid2 is None:
            return 0.0

        mid = (mid1 + mid2) / 2.0
        # 如果开头明显上行（>0.6），中段也至少别变成明显下行（<0.4）
        if head > 0.6 and mid < 0.4:
            return 0.0
        # 否则按差异给分
        return float(np.clip(1.0 - abs(mid - head) / 0.6, 0.0, 1.0))
    
    @staticmethod
    def cadence_motion_reward(grid: np.ndarray) -> float:
        """句末进行奖励：7-1 / 2-1 / 5-1 / 3-2-1（弱一些）"""
        steps, pitches, _ = ClassicalRules._extract_attacks_and_durations(grid)
        if len(pitches) < 2:
            return 0.0

        tonic, mode, _ = ClassicalRules._best_key_major_or_harmonic_minor(grid)
        degs = [ClassicalRules._degree_in_scale(p, tonic, mode) for p in pitches]
        degs = [d for d in degs if d is not None]
        if len(degs) < 2:
            return 0.0

        a, b = degs[-2], degs[-1]
        if (a, b) in [(7, 1), (2, 1), (5, 1)]:
            return 1.0

        # 可选：弱奖励 3-2-1（需要至少3个音级）
        if len(degs) >= 3 and tuple(degs[-3:]) == (3, 2, 1):
            return 0.8

        return 0.0
    

