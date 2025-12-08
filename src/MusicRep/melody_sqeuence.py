import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Note, Instrument, Marker
from miditoolkit.midi.parser import MidiFile
import tempfile
import os

# ==========================================
# 1. 静态配置类 (Configuration)
# ==========================================
class MusicConfig:
    """
    存储乐理常数和约束条件
    """
    # 题目要求的音域 S: F3(53) ~ G5(79)
    PITCH_MIN = 53
    PITCH_MAX = 79
    
    # 编码定义
    REST_VAL = 0   # 休止符
    HOLD_VAL = 1   # 延音符
    
    # 时间结构: 4/4拍, 4小节
    # 最小单位为八分音符 (1拍=2个单位)
    BARS = 4
    BEATS_PER_BAR = 4
    STEPS_PER_BEAT = 2  # 八分音符精度
    TOTAL_STEPS = BARS * BEATS_PER_BAR * STEPS_PER_BEAT # 32
    
    # MIDI 渲染参数
    TICKS_PER_BEAT = 480
    TEMPO = 120

# ==========================================
# 2. 遗传算法用的音乐旋律表示类 (MelodySequence)
# ==========================================
class MelodySequence:
    def __init__(self, grid_data=None):
        """
        Args:
            grid_data (np.array): 长度为 32 的整数数组。
                                  如果为 None，则创建一个全休止的空序列。
        """
        self.config = MusicConfig()
        if grid_data is not None:
            # 确保数据符合约束
            self.grid = np.array(grid_data, dtype=int)
            if len(self.grid) != MusicConfig.TOTAL_STEPS:
                raise ValueError(f"Grid length must be {MusicConfig.TOTAL_STEPS}")
        else:
            self.grid = np.full(MusicConfig.TOTAL_STEPS, MusicConfig.REST_VAL)

    @classmethod
    def from_random(cls):
        """工厂方法：随机生成一个符合音域的序列（用于GA初始化）"""
        new_grid = []
        for _ in range(MusicConfig.TOTAL_STEPS):
            r = np.random.random()
            if r < 0.15: val = MusicConfig.REST_VAL # 有15%概率是休止，直接为0
            elif r < 0.5: val = MusicConfig.HOLD_VAL # 有35%概率是延音, 为1
            else: val = np.random.randint(MusicConfig.PITCH_MIN, MusicConfig.PITCH_MAX + 1) # 50%概率是要求的音域内音符
            new_grid.append(val)
        return cls(new_grid)

    # ==============================
    # 转换 A: 转为 MIDI 对象 (用于保存/播放)
    # ==============================
    def to_midi_object(self, velocity=100):
        """
        将 Grid 数组解码为 miditoolkit 的 MidiFile 对象
        逻辑：合并连续的 HOLD，生成长音符
        """
        midi_obj = MidiFile()
        midi_obj.ticks_per_beat = MusicConfig.TICKS_PER_BEAT
        
        # 创建钢琴轨道
        track = Instrument(program=0, is_drum=False, name="GA Melody")
        
        step_ticks = MusicConfig.TICKS_PER_BEAT // MusicConfig.STEPS_PER_BEAT
        
        current_pitch = None
        current_start_step = 0
        current_duration_steps = 0

        # 遍历 Grid，解析音符
        for i, token in enumerate(self.grid):
            # 情况1: 遇到新音符
            if token > MusicConfig.HOLD_VAL:
                # 结算上一个音符
                if current_pitch is not None:
                    start = current_start_step * step_ticks
                    end = (current_start_step + current_duration_steps) * step_ticks
                    track.notes.append(Note(velocity, current_pitch, start, end))
                
                # 开启新音符
                current_pitch = token
                current_start_step = i
                current_duration_steps = 1
            
            # 情况2: 遇到延音 (HOLD)
            elif token == MusicConfig.HOLD_VAL:
                if current_pitch is not None:
                    current_duration_steps += 1
            
            # 情况3: 遇到休止 (REST)
            elif token == MusicConfig.REST_VAL:
                # 结算上一个音符
                if current_pitch is not None:
                    start = current_start_step * step_ticks
                    end = (current_start_step + current_duration_steps) * step_ticks
                    track.notes.append(Note(velocity, current_pitch, start, end))
                    current_pitch = None # 重置

        # 循环结束后，不要忘了结算最后一个正在响的音
        if current_pitch is not None:
            start = current_start_step * step_ticks
            end = (current_start_step + current_duration_steps) * step_ticks
            track.notes.append(Note(velocity, current_pitch, start, end))

        midi_obj.instruments.append(track)
        return midi_obj

    def save_midi(self, filename="output.mid"):
        """保存为 MIDI 文件"""
        midi_obj = self.to_midi_object()
        midi_obj.dump(filename)
        print(f"Saved MIDI to {filename}")

    # ==============================
    # 转换 B: 转为 AI Tokens (用于 Transformer)
    # ==============================
    def to_remi_tokens(self):
        """
        将 Grid 转换为简化的 REMI Token 序列 (Bar, Pos, Pitch, Dur)，用向量化思路减少 Python 层循环。
        规则：
        - token>1 视为新起音；随后连续的 1 为延音；遇到 0 或下一次起音时结束上一音。
        - 每个小节开始插入 Bar_x；事件按时间排序输出。
        """
        grid = np.asarray(self.grid, dtype=int)
        n = grid.shape[0]
        step_per_bar = MusicConfig.BEATS_PER_BAR * MusicConfig.STEPS_PER_BEAT  # 8

        # 条件掩码
        is_attack = grid > MusicConfig.HOLD_VAL
        not_hold = grid != MusicConfig.HOLD_VAL

        # 所有“起音”位置
        attack_idx = np.flatnonzero(is_attack)
        if attack_idx.size == 0:
            # 只有休止，仍返回 bar 结构
            bars = [f"Bar_{b}" for b in range((n + step_per_bar - 1) // step_per_bar)]
            return bars

        # 预先列出所有非 HOLD 的索引，用于查找下一个音的终止
        non_hold_idx = np.flatnonzero(not_hold)

        # 计算每个起音的结束位置（下一个非 HOLD 索引，若不存在则到序列末尾）
        # 使用 searchsorted 避免逐元素 while
        ends = np.searchsorted(non_hold_idx, attack_idx, side="right")
        next_non_hold = np.empty_like(attack_idx)
        has_next = ends < non_hold_idx.size
        next_non_hold[has_next] = non_hold_idx[ends[has_next]]
        next_non_hold[~has_next] = n

        durations = next_non_hold - attack_idx

        # 构建 bar 事件与音符事件，再按时间排序合并
        events = []
        # bars：每个小节的起始步
        bar_starts = np.arange(0, n, step_per_bar)
        events.extend([(int(b), 0, [f"Bar_{int(b // step_per_bar)}"]) for b in bar_starts])

        # notes：起音步对应的 Pos/Pitch/Dur
        for start, dur, pitch in zip(attack_idx.tolist(), durations.tolist(), grid[attack_idx].tolist()):
            pos_in_bar = start % step_per_bar
            events.append((start, 1, [f"Pos_{pos_in_bar}", f"Pitch_{pitch}", f"Dur_{dur}"]))

        # 按时间排序，同步保持 bar 在同一时间点优先（priority 0 < 1）
        events.sort(key=lambda x: (x[0], x[1]))

        # 拉平成 token 序列
        tokens = []
        for _, _, toks in events:
            tokens.extend(toks)
        return tokens

    # ==============================
    # 转换 C: 转为 Audio (用于试听)
    # ==============================
    def render_wav(self, output_wav="output.wav", soundfont_path=None):
        """
        渲染为 wav 音频。
        需要安装 'midi2audio' 和 'fluidsynth'。
        如果没有 soundfont，这个函数可能无法工作。
        """
        try:
            from midi2audio import FluidSynth
        except ImportError:
            print("Error: 'midi2audio' not installed. Cannot render WAV directly.")
            return

        # 临时保存 MIDI
        temp_midi = "temp_render.mid"
        self.save_midi(temp_midi)
        
        # 尝试使用默认或指定的 SoundFont
        fs = FluidSynth(sound_font=soundfont_path) if soundfont_path else FluidSynth()
        try:
            fs.midi_to_audio(temp_midi, output_wav)
            print(f"Rendered audio to {output_wav}")
        except Exception as e:
            print(f"FluidSynth error: {e}. Is fluidsynth installed on system?")
        finally:
            if os.path.exists(temp_midi):
                os.remove(temp_midi)

# ==========================================
# 3. 单元测试模块
# ==========================================
if __name__ == "__main__":
    print("Testing MusicRep Module...")
    
    # 1. 随机生成一个旋律
    melody = MelodySequence.from_random()
    print(f"Generated Grid (first 10): {melody.grid[:10]}...")
    
    # 2. 导出 MIDI
    melody.save_midi("test_melody.mid")
    
    # 3. 转换为 AI Token
    tokens = melody.to_remi_tokens()
    print(f"Generated AI Tokens (first 5 events): {tokens[:10]}...")
    
    # 4. 手动构造一个 C 大调音阶测试逻辑准确性
    # C4(60), D4(62), E4(64), F4(65)...
    scale_grid = [60, 1, 62, 1, 64, 1, 65, 1, 
                  67, 1, 69, 1, 71, 1, 72, 1,
                  0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0] # 后面补休止
    manual_melody = MelodySequence(scale_grid)
    manual_melody.save_midi("c_scale.mid")
    print("C Scale MIDI saved.")