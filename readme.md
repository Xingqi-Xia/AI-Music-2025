<!-- omit from toc -->
# 音乐与数学2025课程大作业工作文档

<!-- 我用的Markdown All in One可以自动生成目录，但是需要手动给不需要包含的结构加上omit注释。虽然会让这个文件变得臃肿，但是，忍一下吧 @XXQ -->
<!-- omit from toc -->
## 目录

- [项目概述](#项目概述)
- [任务要求](#任务要求)
- [最新进展](#最新进展)
- [TODO](#todo)
- [快速上手（给不写代码也可能不懂音乐的人）](#快速上手给不写代码也可能不懂音乐的人)
  - [安装环境（假设你用的是Windows电脑）](#安装环境假设你用的是windows电脑)
  - [快速玩transformer生成](#快速玩transformer生成)
  - [我需要知道的基础数据格式](#我需要知道的基础数据格式)
  - [如何给遗传算法添加一条“规则”](#如何给遗传算法添加一条规则)
  - [我想看片段对应的音符含义](#我想看片段对应的音符含义)
  - [想快速试听](#想快速试听)
  - [常见参数](#常见参数)
  - [小贴士](#小贴士)
- [项目结构](#项目结构)
  - [项目结构总体概述](#项目结构总体概述)
  - [MusicRep库](#musicrep库)
    - [核心类和函数](#核心类和函数)
  - [GA 库](#ga-库)
    - [主要类](#主要类)
    - [快速示例](#快速示例)
    - [深度模型对接思路](#深度模型对接思路)
  - [transformer库](#transformer库)
    - [核心代码](#核心代码)
    - [运行/示例](#运行示例)
  - [VAE库](#vae库)
    - [核心代码](#核心代码-1)
    - [运行/示例](#运行示例-1)
- [致谢](#致谢)

## 项目概述

- 这是我们的音乐与数学大作业，我们的主题是遗传算法作曲。
- 我们希望训练一个Transformer模型，帮助我们生成初始种群、进行变异和交配，以及作为评估器。我们希望用一套复杂精密的架构骇死助教。

## 任务要求

本部分直接取自教学网中的作业要求。

**机器作曲·遗传算法**

1. 采用下述方法**之一**产生初始种群：
   a) 从具有相同节拍的若干歌曲、乐曲中选取10\~20个长度相等（例如4小节）的片段。
   b) 随机产生：给定乐音体系$$S=\{F_3,\sharp{F}_3,\dots,\sharp{F}_5,G_5\}$$随机选取$S$中的音级，配以不同的的时值，产生10\~20段4/4拍、4小节的“旋律”，其中音符的最短时值为八分音符。
2. 根据课上介绍，搜索相关参考文献．在任何一个软件平台上实现遗传算法。遗传操作应包括交叉(crossover)、变异(mutation)以及对旋律进行的移调、倒影、逆行变换等。
3. 探索建立适应度函数(fitness function)，用以指导旋律进化的方向。
4. 把初始种群作为遗传算法的输入，对其进行遗传迭代，看是否能够得到*较好*的音乐片段。
5. 真实、客观、准确地描述你所完成的各项工作及得到的实际结果，形成完整的实验报告。着重讨论**适应度函数的选取**对于最终产生旋律的音乐特性之间的联系，以及对算法本身效率的影响。

## 最新进展

- ybSun: 用古典音乐数据集训练了一个Transformer模型, 可以生成比较合理的旋律片段, 测试文件在`src/example_transformer.py`。目前还没有训练完, 但已经可以生成一些还不错的片段了。
- ybSun: VAE训练了一小时, 但收敛性非常烂, 大概要弃用了。
- xqXia: 给出了一个新的GA规则
- ybSun: Transformer模型...（说说你训练了什么）
- 我们在12月15日开了个小会，明确了项目进展和分工合作。
- xqXia: 持续跟进维护说明文档

## TODO

这是我们的待办事项列表，从上往下是优先级顺序：
1. 加强遗传算法本身，让它的遗传和变异操作更有效(修改ga_engine.py)
2. 根据人类的音乐习惯, 设计更好的适应度规则（见下方或者`evaluator.py`）*谁会乐理？*
3. 一个改进变异算法的思路: *对每个新繁殖的片段, 随机遮挡几个, 让transformer填充上*(相当于给他拉到局域最优)，这是著名的模因算法。
4. 给我们的架构起个fancy的名字！
5. 将我们的模型架构绘制成图片！然后插入在本readme文件中！
6. 证明transformer-assisted GA的收敛速度快于原始GA！
7. 说明VAE架构非常难以训练！

...

10086. 写一个页面让人交互式地评估音乐进行强化学习？*(感觉找不到那么多人)*


## 快速上手（给不写代码也可能不懂音乐的人）

### 安装环境（假设你用的是Windows电脑）

1. 安装Python。前往[Python官网](https://www.python.org/downloads/)，点击页面上最显眼的"Download Python 3.xx.x"按钮，随后将页面滑到底，下载Windows Installer (64-bit)文件。双击运行该安装包进行安装。安装时全部选择默认选项，但是一定要勾上“将Python添加到环境变量”的选框。
2. [项目主页](https://github.com/Renko6626/AI-Music-2025)右上角的绿色Code按钮下拉，下载本项目的ZIP文件，然后解压到你的电脑中。
3. 在AI-Music-2025文件夹下，Shift加右键，选择“在此处打开命令窗口”（也可以是powershell窗口）。输入命令`pip install -r requirements.txt`并回车执行，pip将自动安装运行本项目所需的全部依赖包。

### 快速玩transformer生成

- 把.pth文件拖到`./transformer/checkpoints_gpt/`目录下
- 运行`src/example_transformer.py`，按照提示选择模型和操作。
- 听曲子

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


## 项目结构

### 项目结构总体概述

- `MusicRep/` : 音乐表示相关代码, 主要用于旋律的不同编码和转换, 以及把旋律进行播放
    - midi_player.py : MIDI文件播放模块
- `GA/` : 遗传算法相关代码
    - `ga_engine.py`：遗传算法主循环，支持移调/逆行/倒影等音乐算子。可直接替换 `evaluator` 为深度模型。
    - `ga_engine_vae.py`：VAE 驱动的遗传算法引擎。
    - `evaluator.py`：评估器接口与规则库；可直接使用 `RuleBasedEvaluator`，也可挂接深度评估器。
- `transformer/`：包含GPT评估器、模型定义、训练脚本和数据预处理，主要用于生成旋律片段和辅助遗传算法种群初始化。
    - `model.py`：MusicGPT 模型定义。
    - `train.py`：GPT 训练脚本与数据集切片加载器。
    - `gpt_evaluator.py`：将训练好的 MusicGPT 封装为 GA 适用的评估器/生成器，支持 `evaluate()` 批量打分与 `generate()` 续写旋律。
    - `dataset/preprocess.py`：从 MIDI 提取主旋律，生成 GPT 训练数据集。
- `VAE/`：基于 GRU 的变分自编码器与评估
    - `vae_evaluator.py`：加载训练好的 GRU-VAE，计算潜空间风格相似度评分。
    - `model/`：`gru.py` 定义 GRU-VAE 主体。
    - `train/`：数据预处理、模型训练与日志（tensorboard events）存放位置。

### MusicRep库

MusicRep 是一个用于表示和处理旋律的python库，它支持我们将会使用的音乐表示方法(网格用于遗传算法、remi token用于Transformer等)。此外，它还包含了一个简易的MIDI合成器，可以将旋律直接渲染为WAV音频文件。

#### 核心类和函数

- `MusicConfig`：乐理与时间网格常数（音域 F3~G5，4/4 拍×4 小节，八分音符精度，共 32 步）。

- `MelodySequence`(来自`melody_sequence.py`)：
    - 处理遗传算法所用的音乐序列, 数据结构是一个32长度的整数数组。
    - 元素位置代表时间步（八分音符），值代表音高（MIDI 编号）或休止0/延音1。
    - 主要方法：
        - `from_random()`：生成一条符合音域的随机旋律。
        - `to_midi_object()` / `save_midi(path)`：合并延音，导出 `miditoolkit.MidiFile` 或直接存成本地文件。
        - `to_remi_tokens()`：生成简化 REMI token 序列（Bar/Pos/Pitch/Dur），主要用来以后喂给Transformer。
        - `render_wav(output_wav, soundfont_path=None)`：将MIDI序列渲染为音频文件（依赖 `midi2audio`/`fluidsynth`，可选）。
        - `render_staff(output)`：输出五线谱（依赖`music21`且需要配置外部环境，可选）。

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


### GA 库

#### 主要类
- `MusicGeneticOptimizer`（`GA/ga_engine.py`）：面向音乐网格（32 步）的遗传算法优化器。
    - 参数：`pop_size` 种群大小，`n_generations` 迭代轮数，`mutation_rate` 点变异概率，`elite_ratio` 精英保留比例（内部至少保留 1 个精英），`evaluator_model` 可接入深度模型批量评估。
    - 方法：`fit(verbose=True)` 运行进化；`predict()` 返回当前最优解对应的 `MelodySequence`。
    - 细节：交叉/变异后会调用 `fixGrid` 修复不合法片段；移调变异检查音域上下界，避免越界；批量评估接口预留给深度模型。

- `MusicEvaluator` / `RuleBasedEvaluator`（`GA/evaluator.py`）：评估器接口与加权规则评估实现。
    - `add_rule(fn, weight)` 注册规则函数（输入单个 grid，返回分数），支持链式调用。
    - `evaluate(population_grid)` 输入 `(pop_size, 32)` 的 numpy 数组，返回分数数组。

- `BasicRules`（`GA/evaluator.py`）：示例规则（C 大调内音奖励、节奏多样性、平滑音程）。

#### 快速示例
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

#### 深度模型对接思路
- 在 `MusicGeneticOptimizer._calculate_fitness_batch` 中，将种群网格转 REMI/token，再批量送入深度模型，返回分数并转回 numpy。
- 建议批量化评估并运行在 GPU；示例代码已预留伪代码位置。


### transformer库

GPT 自回归模型。可以用于直接生成旋律，也可以用于辅助GA算法，生成初始种群并提供适应度函数。

#### 核心代码

- `model.py`：定义GPT模型
  - 类 `MusicGPT`：模型本体
    - `forward(idx, targets=None)` 前向传播，获得logits和CE损失；
    - `generate(idx, max_new_tokens, temperature, top_k)` 自回归地生成新的token，用于音频序列的延伸。
- `gpt_evaluator.py`：评估GPT模型的好坏
  - 类 `GPTMusicEvaluator`：模型评估器
    - `evaluate(population_grid)` 批量评估大量音频序列的适应度，其中适应度定义为损失的倒数。
    - `get_fitness_score(sequence)` 评估单一音频序列的适应度；
    - `generate(prompt_sequence, ...)` 生成音频序列。
- `train.py`：训练GPT模型
  - `MusicGPTDataset`：数据集；
  - `train(config, resume_path)` ：训练并保存模型
- `dataset/preprocess.py` ：从原始音频文件中读取数据集，并进行数据增强，保存结果为文件。

#### 运行/示例

- 数据预处理：`python src/transformer/dataset/preprocess.py`
- 训练：`python src/transformer/train.py --resume latest`（可不带 `--resume`）
- 评估/生成：
```python
from transformer.gpt_evaluator import GPTMusicEvaluator
eva = GPTMusicEvaluator("./transformer/checkpoints_gpt/music_gpt_v1_best.pth")
scores = eva.evaluate(pop_grid)  # pop_grid: [B, T]
new_seq = eva.generate([130], max_new_tokens=128, temperature=1.0, top_k=20)
```

### VAE库
GRU 版离散序列 VAE，学习旋律潜空间并用“风格距离”给 GA 打分，附带 Transformer-VAE 备选实现。

#### 核心代码
- `model/gru.py`和`train/preprocess.py`：数据预处理。读取一个MIDI文件，生成增强后的数据，保存到文件。
- `train/model.py`：定义了各式各样的VAE模型
- `train/dataset.py`：只是下载和保存数据集
- `train/train.py`：非常复杂的训练流程，总之可以训练一个VAE模型出来
- `train/vae_evaluator.py`：对接`GA/evaluator.py/MusicEvaluator`基类，对一段音符片段进行评估。强烈建议将本文件中的`MusicEvaluator`改名，并显式地继承所有evaluator的基类！

#### 运行/示例
- 下载+解压 MIDI：`python src/VAE/train/preprocess.py`
- 生成 32 长度数据集：`python src/VAE/train/dataset.py`
- 训练：`python src/VAE/train/train.py`
- 评估打分：
```python
from VAE.vae_evaluator import MusicEvaluator
import torch
eva = MusicEvaluator("checkpoints/vae_gru_bach_v2_best.pth")
eva.set_target_style(torch.load("classical_dataset.pt")[:1024])
score = eva.get_style_fitness(your_seq)
```

## 致谢

- 感谢ybSun撰写项目代码主体的贡献；
- 感谢ybSun课题组为模型训练提供算力支持；
- 感谢ZZK的火腿肠对ybSun的精神支持；
- 感谢全体组员容忍我写出如此抽象的致谢skr\~。
