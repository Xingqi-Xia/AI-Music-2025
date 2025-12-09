# 音乐与数学2025课程大作业工作文档

## 目录
- [项目概述](#项目概述)
- [快速上手（给懂音乐但不写代码的人）](#快速上手给懂音乐但不写代码的人)
- [MusicRep库](#musicrep库)
- [GA 库](#ga-库)

## 项目概述

这是我们的音乐与数学大作业，我们的主题是遗传算法作曲。
但我想要玩点卷的，我要训练一个Transformer模型来帮助我生成初始种群，或者训练VAE来做评估器。

## 最新进展

- ybSun: 用古典音乐数据集训练了一个Transformer模型, 可以生成比较合理的旋律片段, 测试文件在`src/example_transformer.py`。目前还没有训练完, 但已经可以生成一些还不错的片段了。
- ybSun: VAE训练了一小时, 但收敛性非常烂, 大概要弃用了。
## TODO
这是我们的待办事项列表，从上往下是优先级顺序：
1. 加强遗传算法本身，让它的遗传和变异操作更有效(修改ga_engine.py)
2. 根据人类的音乐习惯, 设计更好的适应度规则(见下方或者evaluator.py)

3. 一个思路: **对每个新繁殖的片段, 随机遮挡几个, 让transformer填充上**(相当于给他拉到局域最优)，这是著名的模因算法。

10086. 写一个页面让人交互式地评估音乐进行强化学习？(感觉找不到那么多人)

## 快速玩transformer生成

- 把.pth文件拖到`./transformer/checkpoints_gpt/`目录下
- 运行`src/example_transformer.py`，按照提示选择模型和操作。
- 听曲子

## 快速上手（给懂音乐但不写代码的人）

### 我需要知道的基础数据格式
- 旋律用一个长度 32 的“网格数组”表示，每格 = 八分音符。
- 数组里的值含义：`0`=休止；`1`=延音；`>1`=MIDI 音高（例如 60 是中央 C）。
- 时间结构固定为 4 小节、4/4 拍、八分音符精度（每小节 8 格，共 32 格）。

### 如何给遗传算法添加一条“规则”
1) 写一个函数, 这个函数应该接受一个数组（形状为 `(32,)` 的 numpy 数组），返回一个浮点数分数（越高越好）。例如在 `GA/evaluator.py` 里添加：
```python
def prefer_steps(grid: np.ndarray) -> float:
    """更喜欢级进（小于等于全音的音程）。"""
    notes = grid[grid > 1]
    if len(notes) < 2:
        return 0.0
    small = np.sum(np.abs(np.diff(notes)) <= 2)
    # 分数归一化到0到1之间
    return small / (len(notes) - 1)
```

**做个文明人，请尽量保证你的分数是0到1之间的数**

2) 在你的脚本里把这条规则加进去并给权重，例如在 `example_ga.py`：
```python
from GA.evaluator import RuleBasedEvaluator, BasicRules

rb = RuleBasedEvaluator()\
    .add_rule(BasicRules.pitch_in_key_c_major, weight=1.0)\
    .add_rule(BasicRules.rhythmic_variety, weight=0.5)\
    .add_rule(BasicRules.smooth_contour, weight=0.5)\
    .add_rule(prefer_steps, weight=0.8)  # 新规则
```
3) 运行 `example_ga.py` 或你的脚本，算法会用新规则打分进化。

### 我想看片段对应的音符含义
- 取一段数组，比如 `[60, 1, 62, 1, 64, 1, 0, 0, ...]`：
  - 第 1 格：60 = C4 起音；第 2 格：1 = 延音；
  - 第 3 格：62 = D4 起音；第 4 格：延音；
  - 第 5 格：64 = E4 起音；第 6 格：延音；
  - 第 7-8 格：0 = 休止两格。
- 每 8 格是一小节，你可以按 8 的间隔来理解分句。

### 想快速试听
- 跑完 GA 后用 `best.save_midi("best.mid")` 生成 MIDI，丢进你熟悉的 DAW 即可。
- 或者用 `Synthesizer.render(best.grid, bpm=120, output_path="best.wav")` 直接生成 WAV（合成器音色可在 `synthesizer.py` 里切换）。


### 常见参数
- 网格长度：32（4 小节 × 4 拍 × 八分音符）。
- 休止/延音：0=Rest，1=Hold。新的音高会终止上一个音并重新起音。
- `MIN_CROP_SIZE`（合成器内部使用的切片时长）：通过持续步数控制；在合成器中自动加 0.2s 尾音让衔接更自然。

### 小贴士
- 你可以在`example_midi.py`中找到如何使用 `MelodySequence` 和 `Synthesizer` 的示例代码。
- 如果只需快速试听，可优先使用 `Synthesizer.render`，它不依赖外部 soundfont。
- 使用 `StringStrategy` 时可调节 `decay_factor` 让拨弦衰减更慢/更快。
- 生成 MIDI 后想用 DAW（如 Ableton/Logic）细修，可直接导入 `*.mid` 文件。




## MusicRep库

MusicRep 是一个用于表示和处理旋律的python库，它支持我们将会使用的音乐表示方法(网格用于遗传算法、remi token用于Transformer等)。此外，它还包含了一个简易的MIDI合成器，可以将旋律直接渲染为WAV音频文件。

### 核心类和函数

- `MusicConfig`：乐理与时间网格常数（音域 F3~G5，4/4 拍×4 小节，八分音符精度，共 32 步）。

- `MelodySequence`(来自`melody_sequence.py`)：
    - 处理遗传算法所用的音乐序列, 数据结构是一个32长度的整数数组。
    - 元素位置代表时间步（八分音符），值代表音高（MIDI 编号）或休止0/延音1。
    - 主要方法：
        - `from_random()`：生成一条符合音域的随机旋律。
        - `to_midi_object()` / `save_midi(path)`：合并延音，导出 `miditoolkit.MidiFile` 或直接存成本地文件。
        - `to_remi_tokens()`：生成简化 REMI token 序列（Bar/Pos/Pitch/Dur），主要用来以后喂给Transformer。
        - `render_wav(output_wav, soundfont_path=None)`：经 MIDI→音频渲染（依赖 `midi2audio`/`fluidsynth`，可选）。

- `Synthesizer`（来自`synthesizer.py`）：一个纯 Python 简易合成器，因为服务器没有声卡，按网格渲染 WAV，不依赖 MIDI 播放。
    - 指定合成音色：
        - 在创建时传入 `strategy` 参数，选择不同的合成策略（如正弦波、方波、拨弦等）。
    - 可选策略：
        - `SineStrategy`：正弦波合成。
        - `SquareStrategy`：方波合成。
        - `StringStrategy`：简易拨弦合成，模拟吉他/弦乐器音色。
    - 主要方法：
        - `render(grid_sequence, bpm=120, output_path="output.wav")`：将 0/1/音高网格直接合成为 WAV。

- `fixGrid(grid_sequence)`：辅助函数，修正输入网格中的不合基本语法的音符（如孤儿hold音符，rest之后的hold等）。


## GA 库

### 主要类
- `MusicGeneticOptimizer`（`GA/ga_engine.py`）：面向音乐网格（32 步）的遗传算法优化器。
    - 参数：`pop_size` 种群大小，`n_generations` 迭代轮数，`mutation_rate` 点变异概率，`elite_ratio` 精英保留比例（内部至少保留 1 个精英），`evaluator_model` 可接入深度模型批量评估。
    - 方法：`fit(verbose=True)` 运行进化；`predict()` 返回当前最优解对应的 `MelodySequence`。
    - 细节：交叉/变异后会调用 `fixGrid` 修复不合法片段；移调变异检查音域上下界，避免越界；批量评估接口预留给深度模型。

- `MusicEvaluator` / `RuleBasedEvaluator`（`GA/evaluator.py`）：评估器接口与加权规则评估实现。
    - `add_rule(fn, weight)` 注册规则函数（输入单个 grid，返回分数），支持链式调用。
    - `evaluate(population_grid)` 输入 `(pop_size, 32)` 的 numpy 数组，返回分数数组。

- `BasicRules`（`GA/evaluator.py`）：示例规则（C 大调内音奖励、节奏多样性、平滑音程）。

### 快速示例
```python
from GA.ga_engine import MusicGeneticOptimizer
from GA.evaluator import RuleBasedEvaluator, BasicRules

rb = RuleBasedEvaluator()\
        .add_rule(BasicRules.pitch_in_key_c_major, weight=1.0)\
        .add_rule(BasicRules.rhythmic_variety, weight=0.5)\
        .add_rule(BasicRules.smooth_contour, weight=0.5)

ga = MusicGeneticOptimizer(pop_size=50, n_generations=20, mutation_rate=0.1, elite_ratio=0.1)
ga.evaluator = rb
ga.fit(verbose=True)

best = ga.predict()
best.save_midi("ga_best.mid")
```

### 深度模型对接思路
- 在 `MusicGeneticOptimizer._calculate_fitness_batch` 中，将种群网格转 REMI/token，再批量送入深度模型，返回分数并转回 numpy。
- 建议批量化评估并运行在 GPU；示例代码已预留伪代码位置。


### TODO

- 2. 根据人的音乐审美习惯, 设计适合的适应度函数。

- 3. 引入深度学习...卷死其他人

### 项目结构

- `MusicRep/` : 音乐表示相关代码, 主要用于旋律的不同编码和转换, 以及把旋律进行播放
    - midi_player.py : MIDI文件播放模块
- `GA/` : 遗传算法相关代码