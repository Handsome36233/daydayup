import numpy as np
import time

np.random.seed(0)

def hungarian_algorithm(cost_matrix):
    """
    使用匈牙利算法解决最小化代价的指派问题。
    
    参数:
        cost_matrix: 输入的 n x m 代价矩阵
    返回:
        最优匹配的总代价和匹配方案
    """
    cost_matrix = np.array(cost_matrix)
    origin_cost_matrix = cost_matrix.copy()
    n, m = cost_matrix.shape
    
    # 步骤 1：行操作 - 每行减去该行的最小值
    for i in range(n):
        cost_matrix[i] -= np.min(cost_matrix[i])
    
    # 步骤 2：列操作 - 每列减去该列的最小值
    for j in range(m):
        cost_matrix[:, j] -= np.min(cost_matrix[:, j])
    
    # 记录标记的星号和质点位置
    star_matrix = np.zeros_like(cost_matrix, dtype=bool)
    prime_matrix = np.zeros_like(cost_matrix, dtype=bool)
    row_covered = np.zeros(n, dtype=bool)
    col_covered = np.zeros(m, dtype=bool)
    
    # 步骤 3：标记每一行中的第一个零，并标记为星号
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                star_matrix[i, j] = True
                row_covered[i] = True
                col_covered[j] = True
    
    # 重置覆盖行和列的标记
    row_covered[:] = False
    col_covered[:] = False
    
    # 步骤 4：覆盖每一列中有星号的列
    def cover_columns_with_stars():
        for i in range(n):
            for j in range(m):
                if star_matrix[i, j]:
                    col_covered[j] = True

    cover_columns_with_stars()
    
    def find_zero():
        """
        找到第一个未覆盖的0元素
        """
        for i in range(n):
            if not row_covered[i]:
                for j in range(m):
                    if cost_matrix[i, j] == 0 and not col_covered[j]:
                        return i, j
        return None, None

    def find_star_in_row(row):
        """
        找到某行中标有星号的列
        """
        for j in range(m):
            if star_matrix[row, j]:
                return j
        return None

    def find_prime_in_row(row):
        """
        找到某行中标有质点的列
        """
        for j in range(m):
            if prime_matrix[row, j]:
                return j
        return None

    def augment_path(path):
        """
        反转路径上的星号和质点
        """
        for r, c in path:
            star_matrix[r, c] = not star_matrix[r, c]

    def adjust_cost_matrix():
        """
        调整代价矩阵以创建更多的零
        """
        min_uncovered_value = np.min(cost_matrix[~row_covered][:, ~col_covered])
        for i in range(n):
            if row_covered[i]:
                cost_matrix[i] += min_uncovered_value
        for j in range(m):
            if not col_covered[j]:
                cost_matrix[:, j] -= min_uncovered_value
    
    while np.sum(col_covered) < min(n, m):
        row, col = find_zero()
        if row is None:  # 没有未覆盖的零，调整代价矩阵
            adjust_cost_matrix()
            row, col = find_zero()

        prime_matrix[row, col] = True
        star_col = find_star_in_row(row)
        if star_col is None:
            # 增广路径
            path = [(row, col)]
            while True:
                star_row = None
                for r in range(n):
                    if star_matrix[r, path[-1][1]]:
                        star_row = r
                        break
                if star_row is None:
                    break
                path.append((star_row, path[-1][1]))

                prime_col = find_prime_in_row(path[-1][0])
                path.append((path[-1][0], prime_col))
            
            augment_path(path)
            prime_matrix[:] = False
            row_covered[:] = False
            col_covered[:] = False
            cover_columns_with_stars()
        else:
            row_covered[row] = True
            col_covered[star_col] = False

    # 计算总代价并返回匹配
    total_cost = 0
    result = []
    for i in range(n):
        for j in range(m):
            if star_matrix[i, j]:
                result.append((i, j))
                total_cost += origin_cost_matrix[i, j]
    return total_cost, result

# 示例使用
if __name__ == "__main__":
    # 输入的 n x m 代价矩阵
    cost_matrix = [
        [12, 7, 9, 7, 9],
        [8, 9, 6, 6, 6],
        [7, 17, 12, 14, 9],
        [15, 14, 6, 6, 10],
        [4, 10, 7, 10, 8]
    ]
    # 生成随机的非方阵代价矩阵
    cost_matrix = np.random.randint(100, 10001, size=(30, 30))
    
    # 调用匈牙利算法
    t1 = time.time()
    total_cost, matching = hungarian_algorithm(cost_matrix)
    t2 = time.time()
    print("运行时间:", t2 - t1, "秒")
    
    print("最小总代价:", total_cost)
    print("匹配关系:", matching)
