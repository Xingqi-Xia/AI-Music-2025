"""
将文本格式的“简谱”转换为音频array序列。
文本格式：
1~7：do~si的音高
0：休止符
-：延音符
L: 低八度符号，可叠加
H: 高八度符号，可叠加
#: 前置升半音符号
b: 前置降半音符号
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
    i = 0
    while i < len(jianpu):
        char = jianpu[i]
        if char == "-":
            seq.append(1)
            i += 1
        elif char == "L":
            offset -= 12
            i += 1
        elif char == "H":
            offset += 12
            i += 1
        elif char == "#":
            if i + 1 < len(jianpu) and jianpu[i + 1].isdigit():
                note = int(jianpu[i + 1])
                if note in PLUSES:
                    pitch = base_pitch + PLUSES[note] + offset + 1
                    seq.append(pitch)
                i += 2
            else:
                i += 1
        elif char == "b":
            if i + 1 < len(jianpu) and jianpu[i + 1].isdigit():
                note = int(jianpu[i + 1])
                if note in PLUSES:
                    pitch = base_pitch + PLUSES[note] + offset - 1
                    seq.append(pitch)
                i += 2
            else:
                i += 1
        elif char.isdigit():
            note = int(char)
            if note == 0:
                seq.append(0)
            elif note in PLUSES:
                pitch = base_pitch + PLUSES[note] + offset
                seq.append(pitch)
            i += 1
        else:
            i += 1
    return fixGrid(seq)