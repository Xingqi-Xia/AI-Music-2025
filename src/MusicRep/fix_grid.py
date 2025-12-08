
# 用于修复旋律基因序列中的语法错误
def fixGrid(grid_array):
    """
    修复旋律序列中的语法错误。
    规则：
    1. '1' (Hold) 不能出现在序列开头。
    2. '1' (Hold) 的前一个位置不能是 '0' (Rest) 
    3. 简化的逻辑是：如果当前是 1，但前一个不是 Pitch 或 1，则把当前强制改为 0 (Rest) 或新的 Pitch。
    """
    fixed_grid = grid_array.copy()
    
    # 规则 1: 首位如果是 Hold，强制变为 Rest (或者随机生成一个 Note)
    if fixed_grid[0] == 1: # 假设 1 是 MusicConfig.HOLD
        fixed_grid[0] = 0  # 变成 Rest，比较安全
    
    # 规则 2: 遍历后续位置
    for i in range(1, len(fixed_grid)):
        if fixed_grid[i] == 1: # 当前是 Hold
            prev = fixed_grid[i-1]
            
            # 如果前一个是 Rest(0)，你不能 Hold 一个 Rest
            if prev == 0:
                fixed_grid[i] = 0 # 也就跟着变成 Rest
            
            # 如果前一个是 Hold(1)，这没问题，构成连音
            # 但如果是一串 [0, 1, 1, 1]，会被上面的逻辑逐步修正为 [0, 0, 1, 1] -> [0, 0, 0, 1] -> [0, 0, 0, 0]
            # 所以只要简单的线性扫描即可解决连环错误。
            
    return fixed_grid