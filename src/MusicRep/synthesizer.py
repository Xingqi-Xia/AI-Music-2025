import numpy as np
from scipy.io import wavfile
from abc import ABC, abstractmethod

# 抽象类 音色合成器的通用接口
class ISoundStrategy(ABC):
    """
    音色合成器接口。
    任何新的乐器生成器都必须继承此类并实现 generate_wave 方法。
    """
    @abstractmethod
    def generate_wave(self, freq: float, duration_sec: float, sample_rate: int) -> np.ndarray:
        """
        根据频率和时长生成一段波形数组。
        """
        pass


class StringStrategy(ISoundStrategy):
    """
    【物理建模】Karplus-Strong 算法。
    模拟拨弦乐器（吉他/古琴/古钢琴）。
    """
    def __init__(self, decay_factor=0.996):
        self.decay_factor = decay_factor

    def generate_wave(self, freq, duration_sec, sample_rate):
        if freq <= 0: return np.zeros(int(sample_rate * duration_sec))
        
        N = int(sample_rate * duration_sec)
        period = int(sample_rate / freq)
        
        # 激励信号 (Burst)
        burst_len = min(period, N)
        burst = np.random.randn(burst_len)
        
        # 初始缓冲
        output = np.zeros(N)
        output[:burst_len] = burst
        
        # 反馈循环 (Feedback Loop)
        for i in range(period, N):
            # 简单的低通滤波器: 平均当前和前一个样本
            output[i] = self.decay_factor * 0.5 * (output[i - period] + output[i - period - 1])
            
        return output

class SineStrategy(ISoundStrategy):
    """
    【加法合成】正弦波 + 泛音 + ADSR 包络。
    听起来像简单的电子琴或风琴。
    """
    def generate_wave(self, freq, duration_sec, sample_rate):
        if freq <= 0: return np.zeros(int(sample_rate * duration_sec))

        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        
        # 叠加几个正弦波让声音厚一点
        wave = 0.6 * np.sin(2 * np.pi * freq * t)
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t) # 二次泛音
        wave += 0.1 * np.sin(2 * np.pi * freq * 3 * t) # 三次泛音
        
        # ADSR 包络 (防止噼里啪啦的爆音)
        attack = int(0.02 * sample_rate)
        release = int(0.1 * sample_rate)
        envelope = np.ones_like(wave)
        
        # Attack (渐强)
        if len(envelope) > attack:
            envelope[:attack] = np.linspace(0, 1, attack)
        # Release (渐弱)
        if len(envelope) > release:
            envelope[-release:] = np.linspace(1, 0, release)
            
        return wave * envelope

class SquareStrategy(ISoundStrategy):
    """
    【8-bit 风格】方波。
    听起来像任天堂 FC / NES 游戏机。
    """
    def generate_wave(self, freq, duration_sec, sample_rate):
        if freq <= 0: return np.zeros(int(sample_rate * duration_sec))
        
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        
        # 生成方波: sign(sin(t))
        wave = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
        
        # 简单的 Decay 包络，让它有点打击感
        decay = np.linspace(1, 0.2, len(wave))
        
        return wave * decay


# 3. 合成器主类
class Synthesizer:
    def __init__(self, strategy: ISoundStrategy = None, sample_rate=44100):
        """
        初始化合成器。
        Args:
            strategy: 实现了 ISoundStrategy 的对象。默认为 StringStrategy。
        """
        self.sample_rate = sample_rate
        # 默认使用8bit方波音色
        if strategy is None:
            self.strategy = SquareStrategy()
            print("[Synthesizer] Using default SquareStrategy (8-bit sound).")
        else:
            self.strategy = strategy

    def set_strategy(self, strategy: ISoundStrategy):
        """运行时切换音色"""
        self.strategy = strategy

    def _midi_to_freq(self, midi_note):
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def render(self, grid_sequence, bpm=120, output_path="output.wav"):
        """
        将 Grid 序列渲染为 WAV 文件。
        处理时间线、Hold 逻辑和混音。
        """
        step_duration = 30.0 / bpm  # 八分音符时长
        # 总时长预留一点尾音 (Release tail)
        total_samples = int(len(grid_sequence) * step_duration * self.sample_rate) + self.sample_rate
        mix_buffer = np.zeros(total_samples)

        # 状态机变量
        current_pitch = None
        current_start_step = 0
        current_duration_steps = 0

        # 定义一个内部函数来处理“添加音符”
        def _add_note(pitch, start_step, duration_steps):
            if pitch <= 1: return # 忽略 Rest(0) 和 Hold(1)
            
            freq = self._midi_to_freq(pitch)
            # 稍微加长一点点 duration 用于 Release，让声音连贯
            duration_sec = (duration_steps * step_duration) + 0.2
            
            # 【关键】调用策略生成波形
            wave = self.strategy.generate_wave(freq, duration_sec, self.sample_rate)
            
            # 计算位置并叠加
            start_sample = int(start_step * step_duration * self.sample_rate)
            end_sample = start_sample + len(wave)
            
            if end_sample > len(mix_buffer):
                end_sample = len(mix_buffer)
                wave = wave[:end_sample - start_sample]
            
            mix_buffer[start_sample:end_sample] += wave

        # 解析 Grid
        for i, token in enumerate(grid_sequence):
            # token 0: REST, token 1: HOLD, token > 1: PITCH
            
            if token != 1: # 如果不是 HOLD，说明有新事件（音符或休止）
                # 1. 结算上一个正在响的音
                if current_pitch is not None:
                    _add_note(current_pitch, current_start_step, current_duration_steps)
                
                # 2. 更新状态
                current_pitch = token if token > 1 else None
                current_start_step = i
                current_duration_steps = 1
            else:
                # 如果是 HOLD，延续时长
                if current_pitch is not None:
                    current_duration_steps += 1
        
        # 循环结束，结算最后一个音
        if current_pitch is not None:
            _add_note(current_pitch, current_start_step, current_duration_steps)

        # 后处理：归一化 (Normalization) 防止爆音
        max_val = np.max(np.abs(mix_buffer))
        if max_val > 0:
            # 留一点动态余量，乘 0.9
            mix_buffer = mix_buffer / max_val * 0.9
        
        # 写入文件
        wavfile.write(output_path, self.sample_rate, (mix_buffer * 32767).astype(np.int16))
        print(f"[{self.strategy.__class__.__name__}] Rendered to {output_path}")
        return output_path