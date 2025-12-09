
# 调用MusicRep包的示例代码

from MusicRep import MelodySequence, Synthesizer, SineStrategy, fixGrid, StringStrategy
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

my_synth = Synthesizer(strategy=StringStrategy())
my_synth.render(melody.grid, bpm=120, output_path=os.path.join(SAVE_FOLDER, "random_melody.wav"))
print("已保存合成音频为 random_melody.wav, 其中音色为正弦波")

melody_test=[130, 62, 66, 69, 74, 73, 74, 71, 74, 69, 74, 67, 74, 66, 78, 76, 74, 73, 71, 69, 1, 67, 69, 66, 1, 0, 78, 76, 74, 73, 71, 0, 76, 74, 73, 71, 69, 71, 74, 66, 1, 68, 1, 69, 1, 1, 1, 1, 1, 62, 66, 69, 74, 73, 74, 71, 74, 69, 74, 67, 74, 66, 78, 76, 74, 73, 71, 69, 1, 67, 69, 66, 1, 0, 78, 76, 74, 73, 71, 0, 76, 74, 73, 71, 69, 71, 74, 66, 1, 68, 1, 69, 1, 1, 1, 1, 1, 0, 69, 67, 66, 64, 62, 0, 62, 64, 66, 67, 71, 69, 69, 67, 1, 66, 67, 66, 1, 64, 1, 1, 1, 62, 66, 69, 74, 73, 74, 71]
my_synth.render(melody_test, bpm=120, output_path=os.path.join(SAVE_FOLDER, "generated_melody_73.wav"))
print("已保存合成音频为 test_melody.wav, 其中音色为正弦波")