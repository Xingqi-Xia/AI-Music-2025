
# 调用MusicRep包的示例代码

from MusicRep import MelodySequence, Synthesizer, SineStrategy, fixGrid
import os
# 储存位置
SAVE_FOLDER="example_outputs/musicrep/"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
# 生成一个随机旋律序列
melody = MelodySequence.from_random()
print("随机生成的旋律序列:", melody.grid)

# 用 fixGrid 修复旋律序列中的语法错误
melody.grid = fixGrid(melody.grid)
print("修复后的旋律序列:", melody.grid)

# 将旋律转换为MIDI并保存
melody.save_midi(os.path.join(SAVE_FOLDER, "random_melody.mid"))
print("已保存为 random_melody.mid")


# 试听这段旋律

my_synth = Synthesizer(strategy=SineStrategy())
my_synth.render(melody.grid, bpm=120, output_path=os.path.join(SAVE_FOLDER, "random_melody.wav"))
print("已保存合成音频为 random_melody.wav, 其中音色为正弦波")
