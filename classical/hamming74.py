import numpy as np
from typing import Tuple

class LinearBlockCode:
    """
    简单的线性分组码类，用于演示 Hamming(7,4)。
    仅考虑 0/1 比特，加法在 GF(2) 上进行。
    """

    def __init__(self, G: np.ndarray, H: np.ndarray):
        """
        参数:
            G: (k, n) 生成矩阵
            H: (n-k, n) 校验矩阵
        """
        self.G = G.astype(int) % 2
        self.H = H.astype(int) % 2
        self.k, self.n = self.G.shape

        # 自动生成“综合征 -> 单比特错误向量”的查找表
        self.syndrome_table = self._build_syndrome_table()

    # ------------ 基本线性代数操作 ------------

    @staticmethod
    def _mod2(x: np.ndarray) -> np.ndarray:
        return np.mod(x, 2)

    # ------------ 核心功能：编码 / 综合征 / 解码 ------------

    def encode(self, m: np.ndarray) -> np.ndarray:
        """
        对长度为 k 的消息比特 m 进行编码，得到长度为 n 的码字。

        m: shape = (k,)
        return: shape = (n,)
        """
        m = np.array(m, dtype=int).reshape(1, -1)  # 视作行向量
        c = self._mod2(m @ self.G)                 # (1, n)
        return c.flatten()

    def syndrome(self, y: np.ndarray) -> np.ndarray:
        """
        计算接收向量 y 的综合征 s = y H^T (mod 2)

        y: shape = (n,)
        return: shape = (n-k,)
        """
        y = np.array(y, dtype=int).reshape(1, -1)
        s = self._mod2(y @ self.H.T)
        return s.flatten()

    def decode(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对接收向量 y 进行最简单的“单比特错误纠正”解码：
        1. 计算综合征 s
        2. 查表获得对应的错误向量 e_hat
        3. 得到纠正后的码字 c_hat = y - e_hat
        4. 取前 k 位作为估计的消息比特 m_hat （这里假设系统码形式）

        return: (m_hat, c_hat, s, e_hat)
        """
        y = np.array(y, dtype=int).flatten()
        s = self.syndrome(y)
        key = tuple(s.tolist())

        # 查找对应的估计错误向量
        e_hat = self.syndrome_table.get(key, np.zeros(self.n, dtype=int))

        # 纠正后的码字
        c_hat = self._mod2(y - e_hat)

        # 这里假设 G 是系统码形式：[I_k | P]，所以消息比特在前 k 位
        m_hat = c_hat[:self.k]

        return m_hat, c_hat, s, e_hat 

    # ------------ 构造综合征查找表（单比特错误） ------------

    def _build_syndrome_table(self):
        """
        对所有“0 错误 + 单比特错误”枚举，构造综合征查找表:
            syndrome(tuple) -> error_vector (length n)

        Hamming(7,4) 的最小距离 d=3，只需要考虑单比特错误即可。
        """
        table = {}

        # 情况 1：无错误（综合征为全 0，对应 e = 0）
        zero_error = np.zeros(self.n, dtype=int)
        s0 = tuple(self.syndrome(zero_error).tolist())
        table[s0] = zero_error

        # 情况 2：所有单比特错误 e_i
        for i in range(self.n):
            e = np.zeros(self.n, dtype=int)
            e[i] = 1
            s = tuple(self.syndrome(e).tolist())
            # 如果不同错误给出相同综合征，这里会覆盖，但 Hamming(7,4) 不会
            table[s] = e

        return table


# ------------ 具体的 Hamming(7,4) 码实例 ------------

def create_hamming_74() -> LinearBlockCode:
    """
    构造一个标准的 Hamming(7,4) 码实例。

    这里使用一种常见的系统码表示：
        G = [I_4 | P], H = [P^T | I_3]

    你可以根据自己文档中的定义替换 G, H。
    """

    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=int)

    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ], dtype=int)

    return LinearBlockCode(G, H)


# ------------ 一个小 demo：展示完整的编码 / 加错 / 解码流程 ------------

def add_single_bit_error(c: np.ndarray, position: int) -> np.ndarray:
    """
    在码字 c 的某一位 position 上加入 1 比特错误（按位翻转）。
    position 从 0 开始计数。
    """
    y = np.array(c, dtype=int).copy()
    if 0 <= position < len(y):
        y[position] ^= 1  # 0->1 或 1->0
    return y


def run_demo():
    """
    在命令行运行时，展示一个编码->加错误->综合征->纠错->解码的完整过程。
    """
    code = create_hamming_74()

    # 1. 选择一个 4 比特消息 m
    m = np.array([1, 0, 1, 1], dtype=int)
    print("原始消息 m:", m)

    # 2. 编码得到 7 比特码字 c
    c = code.encode(m)
    print("编码后码字 c:", c)

    # 3. 在第 2 位（从 0 数是 index=1）引入单比特错误
    error_pos = 1
    y = add_single_bit_error(c, error_pos)
    print(f"在位置 {error_pos} 处加入 1 比特错误后的接收向量 y:", y)

    # 4. 计算综合征
    s = code.syndrome(y)
    print("综合征 s = y H^T (mod 2):", s)

    # 5. 用解码器纠错，得到估计的消息与码字
    m_hat, c_hat, s_hat, e_hat = code.decode(y)
    print("解码得到的估计消息 m_hat:", m_hat)
    print("解码得到的估计码字 c_hat:", c_hat)
    print("解码器认为的错误向量 e_hat:", e_hat)

    # 6. 对比结果
    print("消息是否恢复正确？", np.array_equal(m, m_hat))
    print("码字是否恢复正确？", np.array_equal(c, c_hat))


if __name__ == "__main__":
    run_demo()
