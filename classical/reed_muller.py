import numpy as np
from typing import Tuple
from .hamming74 import LinearBlockCode


def build_rm_1_3_G() -> np.ndarray:
    """
    构造 RM(1,3) 的生成矩阵 G。
    
    RM(1,3) 是所有 3 变量一次及以下布尔多项式的值组成的 8 位代码字集合。
    变量按顺序排列为 x1, x2, x3，取值为所有 8 个输入组合：
        (0,0,0)
        (0,0,1)
        (0,1,0)
        (0,1,1)
        (1,0,0)
        (1,0,1)
        (1,1,0)
        (1,1,1)
    
    对应生成矩阵 G 行为：
        1                          常数项
        x1
        x2
        x3
    """
    # 所有 3 比特输入组合
    inputs = np.array(
        [[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)],
        dtype=int
    ).T  # 3 x 8 矩阵，行分别是 x1, x2, x3

    x1 = inputs[0]
    x2 = inputs[1]
    x3 = inputs[2]

    # 生成矩阵 G 的 4 行
    G = np.vstack([
        np.ones(8, dtype=int),  # 常数项 1
        x1,
        x2,
        x3
    ])

    return G % 2


def build_rm_1_3_H(G: np.ndarray) -> np.ndarray:
    """
    构造 RM(1,3) 的校验矩阵 H。
    方法：
        使用 RM(0,3)（常数项）和 RM(1,3) 的层级关系：
            RM(1,3) 的对偶码是 RM(1,3)^⊥ = RM(0,3)
        但更常用的是直接找 G 的 nullspace 作为 H。

    我们直接通过 GF(2) 的线性代数求解 H，使得 H * G^T = 0。
    """
    G = np.array(G, dtype=int) % 2
    k, n = G.shape

    # 在 GF(2) 构造 null space
    # 通过求解 x G^T = 0
    H_list = []

    for i in range(1, 1 << n):
        v = np.array([(i >> bit) & 1 for bit in range(n)], dtype=int)
        if np.all((v @ G.T) % 2 == 0):
            H_list.append(v)

    # 去重并规范形 (rank = n-k = 8-4 = 4)
    # 只保留线性无关向量
    H = []
    for v in H_list:
        if not spans(H, v):
            H.append(v)
        if len(H) == n - k:
            break

    return np.array(H, dtype=int)


def spans(basis, v):
    """
    检查 v 是否可以由 basis 中的向量生成（GF(2) 线性组合）。
    """
    if len(basis) == 0:
        return False
    B = np.array(basis, dtype=int).T  # n x m
    try:
        # 求 x 使得 B x = v (mod 2)
        # 解不出来就说明不在 span 中
        _ = solve_mod2(B, v)
        return True
    except Exception:
        return False


def solve_mod2(A, b):
    """
    解 A x = b (mod 2) 的线性方程，
    仅用于判断可解性（返回某个解即可）。
    """
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape

    # 增广矩阵
    aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)

    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] ^= aug[[pivot, row]]
        for r in range(m):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]
        row += 1

    # 检查是否无解
    for r in range(row, m):
        if aug[r, :-1].sum() == 0 and aug[r, -1] == 1:
            raise ValueError("No solution in GF(2).")

    # 任意取一个解（非唯一）
    x = np.zeros(n, dtype=int)
    return x


# ------------------ 构造 Reed-Muller(1,3) 码 ------------------

def create_rm_1_3() -> LinearBlockCode:
    """
    构造 (8,4,4) Reed-Muller RM(1,3) 码。
    """
    G = build_rm_1_3_G()              # 4 x 8
    H = build_rm_1_3_H(G)             # 4 x 8

    # 返回 LinearBlockCode 实例
    code = LinearBlockCode(G, H)
    return code


# ------------------ 示范 demo ------------------

def run_demo():
    code = create_rm_1_3()

    print("构造了 Reed–Muller RM(1,3) (8,4,4) 码：")
    print("G shape:", code.G.shape)
    print("H shape:", code.H.shape)

    # 随机消息
    rng = np.random.default_rng(0)
    m = rng.integers(0, 2, size=4)
    print("\n原始消息 m:", m)

    # 编码
    c = code.encode(m)
    print("编码后码字 c:", c)

    # 引入错误
    y = c.copy()
    y[3] ^= 1   # 在某一位翻转
    print("加入错误后的 y:", y)

    # 解码
    m_hat, c_hat, s, e_hat = code.decode(y)
    print("\n综合征:", s)
    print("估计错误 e_hat:", e_hat)
    print("解码消息 m_hat:", m_hat)

    print("\n消息恢复正确？", np.array_equal(m, m_hat))
    print("码字恢复正确？", np.array_equal(c, c_hat))


if __name__ == "__main__":
    run_demo()