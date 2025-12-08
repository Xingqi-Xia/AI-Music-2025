from GA import MusicEvaluator, RuleBasedEvaluator, BasicRules, MusicGeneticOptimizer
import numpy as np
from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy
import os

EXAMPLE_PATH="example_outputs/ga_example/"
if not os.path.exists(EXAMPLE_PATH):
    os.makedirs(EXAMPLE_PATH)
example_evaluator = RuleBasedEvaluator()
# 添加一些简单的规则
example_evaluator.add_rule(BasicRules.smooth_contour, weight=0.5)
example_evaluator.add_rule(BasicRules.rhythmic_variety, weight=0.5)
example_evaluator.add_rule(BasicRules.pitch_in_key_c_major, weight=1.0)

def my_custom_rule(grid: np.ndarray) -> float:
    """自定义规则示例：奖励包含更多不同音高的旋律"""
    notes = grid[grid > 1] # 排除 Rest(0) 和 Hold(1)
    unique_notes = len(set(notes))
    return np.tanh(unique_notes / 12.0)  
example_evaluator.add_rule(my_custom_rule, weight=0.8)


print("已设置评估器规则: smooth_contour, rhythmic_variety, my_custom_rule")
#example_evaluator.add_rule(BasicRules.pitch_in_key_c_major, weight=0.3)
ga_optimizer = MusicGeneticOptimizer(
    pop_size=100,
    n_generations=40,
    elite_ratio=0.1,
    prob_point_mutation=0.15,
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

# 输出结果
print("最优旋律序列的音符网格:", best_melody.grid)
# 导出为wav文件试听
synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody.wav"))
print("已保存最优旋律的合成音频为 best_melody.wav")