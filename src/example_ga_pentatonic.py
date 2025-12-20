from GA import MusicEvaluator, RuleBasedEvaluator, BasicRules, MusicGeneticOptimizer, PentatonicRules
import numpy as np
from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy
import os

# 为了结果可复现，添加随机数种子
np.random.seed(int.from_bytes("SYBNB!".encode()[:4],'big'))

EXAMPLE_PATH="example_outputs/ga_example/"
if not os.path.exists(EXAMPLE_PATH):
    os.makedirs(EXAMPLE_PATH)

evaluator = RuleBasedEvaluator()
#这是五声调式，写这个是因为好写一点

# ===== Pentatonic Evaluator =====
evaluator.add_rule(PentatonicRules.pentatonic_fit, weight=2.0)
evaluator.add_rule(PentatonicRules.stepwise_motion_preference, weight=1.4)
evaluator.add_rule(PentatonicRules.melodic_flow, weight=1.0)
evaluator.add_rule(PentatonicRules.contour_variation_reward, weight=0.9)  
evaluator.add_rule(PentatonicRules.overlong_note_penalty, weight=1.3)    
evaluator.add_rule(PentatonicRules.rest_sparsity_penalty, weight=1.1)
evaluator.add_rule(PentatonicRules.note_density_target, weight=1.0)
evaluator.add_rule(PentatonicRules.register_balance, weight=0.8)

ga_optimizer = MusicGeneticOptimizer(
    pop_size=100,
    n_generations=500,
    elite_ratio=0.2,
    prob_point_mutation=0.1,
    prob_transposition=0,
    prob_retrograde=0,
    prob_inversion=0,
    evaluator_model=evaluator,
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
best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody_pentatonic.png"))
# 导出为wav文件试听
synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody_pentatonic.wav"))
print("已保存最优旋律的合成音频为 best_melody_pentatonic.wav")
