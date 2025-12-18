"""
将文本格式的“简谱”转换为音频array序列。
文本格式：
1~7：do~si的音高
0：休止符
-：延音符
L: 低八度符号，单音符有效
H: 高八度符号，单音符有效
#: 前置升半音符号，单音符有效
b: 前置降半音符号，单音符有效
"""

from .fix_grid import fixGrid

def jianpu_to_array(jianpu: str, base_pitch: int = 60) -> list[int]:
    PLUSES = {
        1: 0,
        2: 2,
        3: 4,
        4: 5,
        5: 7,
        6: 9,
        7: 11,
    }
    seq = []
    offset = 0
    accent = 0
    for c in jianpu:
        if c == "-":
            seq.append(1)
        elif c == "L":
            offset -= 12
        elif c == "H":
            offset += 12
        elif c == "#":
            accent += 1
        elif c == "b":
            accent -= 1
        elif c.isdigit():
            note = int(c)
            if note == 0:
                seq.append(0)
            elif note in PLUSES:
                pitch = base_pitch + PLUSES[note] + offset + accent
                seq.append(pitch)
            offset=0
            accent=0
    return fixGrid(seq)