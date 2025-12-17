# -*- coding: utf-8 -*-
"""
基于example_ga.py的简单修改，单凭我个人的感觉添加更多规则。
现在能够输出能听的音乐了！

作者：Xingqi-Xia
创建时间：2025-12-18
版本：1.0.0
"""

from GA import MusicEvaluator, RuleBasedEvaluator, MusicGeneticOptimizer, BasicRules
import numpy as np
from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy
import os
import difflib
from collections import defaultdict

EXAMPLE_PATH="example_outputs/ga_example/"
if not os.path.exists(EXAMPLE_PATH):
    os.makedirs(EXAMPLE_PATH)
example_evaluator = RuleBasedEvaluator()

MusicConfig.BARS=32
# 添加一些不那么简单的规则
example_evaluator.add_rule(BasicRules.smooth_contour, weight=1.0)

# 为了结果可复现，添加随机数种子
np.random.seed(int.from_bytes("SYBNB!".encode()[:4],'big'))

def pitch_in_key_c_major(grid):
    "奖励C大调中的音阶，尤其奖励主音和属音，惩罚副音。"
    score_dict=defaultdict(lambda:0, {
        0:1,
        2:0.5, 
        4:0.7,
        5:0.2,
        7:1, 
        9:0.9,
        11:0.2
    })
    notes = grid[grid > 1] # 排除 Rest(0) 和 Hold(1)
    if len(notes) == 0: return 0.0
    
    count = sum(score_dict[int(n%12)] for n in notes)
    return count / len(notes)

def rhythmic_variety(grid):
    "奖励单一乐句中的节奏和旋律的变化，惩罚过多休止或全屏延音"
    changes = np.sum(grid != 1)
    ratio = changes / len(grid)
    return max(4*(ratio-0.2)**2, 1)


def pitch_variety(grid):
    """奖励包含更多不同音高的旋律"""
    notes = grid[grid > 1] # 排除 Rest(0) 和 Hold(1)
    unique_notes = len(set(notes))
    return np.tanh(unique_notes / 4.0) 

def rhythmic_repetition(grid):
    "奖励乐句间节奏的重复"
    periods=grid.reshape((-1, 32)) # 得到不同的乐句
    # 计算乐句间节奏的重复
    bonus=0
    for i in range(32):
        bonus+=np.sum(np.unique_counts(periods[:,i]).counts-1)
    return min(bonus/periods.size, 1)

def pitch_repetition(grid):
    "奖励乐句间旋律的重复"
    periods=grid.reshape((-1, 32)) # 得到不同的乐句
    # 计算乐句之间旋律的重复
    pitches=[p[p > 1] for p in periods]
    # 给出配对
    pairs=[]
    for _gap_exp in range(int(np.log2(len(periods))), -1, -1):
        gap=2**_gap_exp
        for i in range(len(periods)-gap):
            pairs.append((i, i+gap))
    for p in pairs:
        bonus+=difflib.SequenceMatcher(None, pitches[p[0]], pitches[p[1]]).ratio()
    return bonus/len(pairs)

example_evaluator.add_rule(pitch_in_key_c_major, weight=1.0)
example_evaluator.add_rule(rhythmic_variety, weight=2.0)
example_evaluator.add_rule(pitch_variety, weight=1.0)
example_evaluator.add_rule(rhythmic_repetition, weight=1.0)

ga_optimizer = MusicGeneticOptimizer(
    pop_size=100,
    n_generations=500,
    elite_ratio=0.2,
    prob_point_mutation=0.1,
    prob_transposition=0,
    prob_retrograde=0,
    prob_inversion=0,
    evaluator_model=example_evaluator,
    device='cpu'
)

# 初始化音频生成器
synth = Synthesizer(strategy=SineStrategy())

# 初始化种群
ga_optimizer._initialize_population()

# 运行遗传算法优化
ga_optimizer.fit()
# 获取最优个体
best_melody_grid = ga_optimizer.best_individual_
best_melody = MelodySequence(best_melody_grid)

# 导出为wav文件试听
best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody_2.png"))
synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody_2.wav"))

