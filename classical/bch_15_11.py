import numpy as np
from itertools import combinations
from .hamming74 import LinearBlockCode


# ---------------- GF(2) 工具函数 ----------------

def gf2_inv(A: np.ndarray) -> np.ndarray:
    """
    在 GF(2) 上求一个方阵 A 的逆矩阵。
    若矩阵不可逆，则抛出 ValueError。
    """
    A = np.array(A, dtype=int) % 2
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("gf2_inv 只支持方阵")

    # 构造增广矩阵 [A | I]
    aug = np.concatenate([A.copy(), np.eye(n, dtype=int)], axis=1)

    # 高斯消元
    row = 0
    for col in range(n):
        # 在当前列寻找一个主元（值为 1）
        pivot = None
        for r in range(row, n):
            if aug[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            # 当前列找不到主元，说明矩阵不可逆
            raise ValueError("矩阵在 GF(2) 上不可逆")

        # 把主元行换到当前 row
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]

        # 消去当前列中其他行的 1
        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r, :] ^= aug[row, :]

        row += 1
        if row == n:
            break

    # 此时左半部分应为单位阵，右半部分即为逆矩阵
    inv = aug[:, n:]
    return inv % 2


def build_h_matrix_15_11() -> np.ndarray:
    """
    构造 (15,11) Hamming/BCH 码的一种标准校验矩阵 H (4 x 15)。
    方法：将每一列设置为 1~15 的 4 比特二进制表示（不包含全 0）。
    """
    H = np.zeros((4, 15), dtype=int)

    for j in range(1, 16):  # 列号从 1 到 15
        # 将 j 转为 4 位二进制：b3 b2 b1 b0
        b3 = (j >> 3) & 1
        b2 = (j >> 2) & 1
        b1 = (j >> 1) & 1
        b0 = j & 1
        H[:, j - 1] = np.array([b3, b2, b1, b0], dtype=int)

    return H


def build_systematic_G_from_H(H: np.ndarray):
    """
    给定 H (4 x 15)，通过选择 4 列作为奇偶校验位，构造系统型生成矩阵：
        - 重排列使 H = [A | B]，其中 B 为 4x4 可逆矩阵
        - 计算 P^T = B^{-1} A (在 GF(2) 上)
        - 得到 G = [I_k | P]，k = 11

    返回:
        G_reordered: (11, 15) 生成矩阵
        H_reordered: (4, 15) 重新排好列顺序的校验矩阵
    """
    H = np.array(H, dtype=int) % 2
    rows, n = H.shape
    if rows != 4 or n != 15:
        raise ValueError("当前函数只针对 4x15 的 H 设计")

    # 尝试在 15 列中选出 4 列构成可逆的 B
    cols = list(range(n))
    chosen_parity_cols = None
    for comb in combinations(cols, 4):
        B_candidate = H[:, comb]
        try:
            _ = gf2_inv(B_candidate)
            chosen_parity_cols = list(comb)
            break
        except ValueError:
            continue

    if chosen_parity_cols is None:
        raise RuntimeError("没有找到可逆的 4x4 子矩阵用于构造系统型 G")

    # 剩下的列就是“消息位”列
    chosen_message_cols = [c for c in cols if c not in chosen_parity_cols]

    # 按 [message_cols | parity_cols] 重排 H
    new_order = chosen_message_cols + chosen_parity_cols
    H_reordered = H[:, new_order]

    # 现在 H_reordered = [A | B]
    A = H_reordered[:, :11]   # 4 x 11
    B = H_reordered[:, 11:]   # 4 x 4

    # 计算 P^T = B^{-1} * A  (GF(2) 上)
    B_inv = gf2_inv(B)        # 4 x 4
    P_T = (B_inv @ A) % 2     # 4 x 11
    P = P_T.T                 # 11 x 4

    # 构造 G = [I_11 | P]
    I_k = np.eye(11, dtype=int)
    G_reordered = np.concatenate([I_k, P], axis=1)  # 11 x 15

    return G_reordered, H_reordered


# ---------------- 具体的 BCH(15,11,3) 码构造 ----------------

def create_bch_15_11() -> LinearBlockCode:
    """
    构造一个 BCH(15,11,3) 码（等价于 Hamming(15,11)）的 LinearBlockCode 实例。

    - 长度 n = 15
    - 维数 k = 11
    - 最小距离 d = 3
    - 可纠正 1 比特错误
    """
    # 1. 先构造一个 4x15 的 H，列为 1..15 的二进制表示
    H = build_h_matrix_15_11()

    # 2. 从 H 中构造系统型 G 和重排后的 H
    G_sys, H_sys = build_systematic_G_from_H(H)

    # 3. 用已有的 LinearBlockCode 封装
    code = LinearBlockCode(G_sys, H_sys)
    return code


# ---------------- 一个简单的 demo ----------------

def run_demo():
    """
    演示 BCH(15,11) 的编码/加错/解码流程。
    """
    code = create_bch_15_11()
    print("构造了 BCH(15,11,3) 码：")
    print(f"  生成矩阵 G 形状: {code.G.shape}")
    print(f"  校验矩阵 H 形状: {code.H.shape}")

    # 随机生成一个 11 比特消息
    rng = np.random.default_rng(42)
    m = rng.integers(0, 2, size=(code.k,), dtype=int)
    print("\n原始消息 m:", m)

    # 编码
    c = code.encode(m)
    print("编码后码字 c:", c)

    # 在某一位上引入 1 比特错误（比如第 5 位）
    error_pos = 5
    y = c.copy()
    y[error_pos] ^= 1
    print(f"\n在位置 {error_pos} 处引入 1 比特错误后的接收向量 y:", y)

    # 解码
    m_hat, c_hat, syndrome, e_hat = code.decode(y)
    print("\n综合征 syndrome:", syndrome)
    print("解码器估计的错误向量 e_hat:", e_hat)
    print("解码得到的消息 m_hat:", m_hat)
    print("解码得到的码字 c_hat:", c_hat)

    print("\n消息是否恢复正确？", np.array_equal(m, m_hat))
    print("码字是否恢复正确？", np.array_equal(c, c_hat))


if __name__ == "__main__":
    run_demo()
