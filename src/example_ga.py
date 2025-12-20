from GA import MusicEvaluator, RuleBasedEvaluator, BasicRules, MusicGeneticOptimizer
import numpy as np
from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy
import os

# 为了结果可复现，添加随机数种子
np.random.seed(int.from_bytes("SYBNB!".encode()[:4],'big'))

EXAMPLE_PATH="example_outputs/ga_example/"
if not os.path.exists(EXAMPLE_PATH):
    os.makedirs(EXAMPLE_PATH)
example_evaluator = RuleBasedEvaluator()
"""
我把这一段也注释掉了虽然它的确没什么用去了也行
# 添加一些简单的规则
example_evaluator.add_rule(BasicRules.smooth_contour, weight=1.0)
example_evaluator.add_rule(BasicRules.rhythmic_variety, weight=0.5)
example_evaluator.add_rule(BasicRules.pitch_in_key_c_major, weight=1.0)

def my_custom_rule(grid: np.ndarray) -> float:
    """自定义规则示例：奖励包含更多不同音高的旋律"""
    notes = grid[grid > 1] # 排除 Rest(0) 和 Hold(1)
    unique_notes = len(set(notes))
    return np.tanh(unique_notes / 12.0)  
example_evaluator.add_rule(my_custom_rule, weight=0.8)
"""

""" 
#这一部分是古典的，我尽力了，我的乐理水平就到这了（要跑这个就把注释去掉然后把后面那个五声调式注释掉）

# ===== Layer A =====
example_evaluator.add_rule(ClassicalRules.key_fit_best_of_24, weight=1.4)
example_evaluator.add_rule(ClassicalRules.interval_stepwise_preference, weight=1.2)
example_evaluator.add_rule(ClassicalRules.rest_and_long_rest_penalty, weight=1.3)
example_evaluator.add_rule(ClassicalRules.melodic_range_score, weight=0.6)

# ===== Layer B =====
example_evaluator.add_rule(ClassicalRules.phrase_start_on_strongbeat, weight=0.6)
example_evaluator.add_rule(ClassicalRules.cadence_end_stable_and_long, weight=1.6)
example_evaluator.add_rule(ClassicalRules.motif_repetition_interval, weight=0.9)
example_evaluator.add_rule(ClassicalRules.syncopation_penalty, weight=1.1)
example_evaluator.add_rule(ClassicalRules.chromatic_semitone_overuse_penalty, weight=0.6)

# ===== Layer C =====
example_evaluator.add_rule(ClassicalRules.note_density_target, weight=1.0)
example_evaluator.add_rule(ClassicalRules.leading_tone_resolution, weight=0.9)

 """

# 暂时移除（这个不要动，完成版的就没有）
# ClassicalRules.primary_degree_ratio
# ClassicalRules.climax_position_score
# ClassicalRules.turning_points_target
# ClassicalRules.leading_tone_resolution
# ClassicalRules.dotted_rhythm_reward
# ClassicalRules.pre_barline_rest_reward

#这是五声调式，写这个是因为好写一点

# ===== Pentatonic Evaluator =====

# ===== Pentatonic Evaluator (Enhanced) =====

example_evaluator.add_rule(PentatonicRules.pentatonic_fit, weight=2.0)

example_evaluator.add_rule(PentatonicRules.stepwise_motion_preference, weight=1.4)
example_evaluator.add_rule(PentatonicRules.melodic_flow, weight=1.0)

example_evaluator.add_rule(PentatonicRules.contour_variation_reward, weight=0.9)  
example_evaluator.add_rule(PentatonicRules.overlong_note_penalty, weight=1.3)    

example_evaluator.add_rule(PentatonicRules.rest_sparsity_penalty, weight=1.1)
example_evaluator.add_rule(PentatonicRules.note_density_target, weight=1.0)
example_evaluator.add_rule(PentatonicRules.register_balance, weight=0.8)


print("已设置评估器规则: smooth_contour, rhythmic_variety, my_custom_rule")
#example_evaluator.add_rule(BasicRules.pitch_in_key_c_major, weight=0.3)
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

# 输出结果
print("最优旋律序列的音符网格:", best_melody.grid)
best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody.png"))
# 导出为wav文件试听
synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody.wav"))
print("已保存最优旋律的合成音频为 best_melody.wav")
