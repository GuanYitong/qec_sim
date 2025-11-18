import numpy as np
from typing import List, Tuple, Optional

class RepetitionCode:
    """
    经典重复码 (n, 1, n)，只编码单比特消息 m ∈ {0,1}：
        m = 0 -> 000...0
        m = 1 -> 111...1

    解码使用多数表决（majority vote）。
    """

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("重复码长度 n 必须为正整数")
        self.n = n
        self.k = 1  # 只编码 1 个消息比特

    # ------------ 编码 ------------

    def encode(self, m: int) -> np.ndarray:
        """
        对单比特消息 m 编码为长度 n 的码字。
        m: 0 或 1
        return: 长度 n 的 numpy 数组
        """
        if m not in (0, 1):
            raise ValueError("消息 m 必须是 0 或 1")
        return np.full(self.n, m, dtype=int)

    # ------------ 噪声 / 错误注入 ------------

    def add_errors(self, c: np.ndarray, error_positions: List[int]) -> np.ndarray:
        """
        在码字 c 指定的位置上翻转比特。

        参数:
            c: 原始码字，长度 n
            error_positions: 需要翻转的比特下标列表（从 0 开始）
        """
        y = np.array(c, dtype=int).copy()
        for pos in error_positions:
            if 0 <= pos < self.n:
                y[pos] ^= 1
        return y


    
    def add_random_noise(self, c: np.ndarray, p: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Rest of your function implementation here
        # Use rng for random number generation
        """
        对码字 c 中的每一位，以概率 p 发生翻转（独立同分布）。

        参数:
            c: 原始码字
            p: 单比特翻转概率 (0 <= p <= 1)
            rng: 可选的随机数生成器
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("错误概率 p 必须在 [0, 1] 区间内")

        if rng is None:
            rng = np.random.default_rng()

        y = np.array(c, dtype=int).copy()
        flips = rng.random(self.n) < p  # True 表示这一位发生翻转
        y[flips] ^= 1
        return y

    # ------------ 解码（多数表决） ------------

    def decode(self, y: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        对接收向量 y 进行解码：
            - 多数表决得到估计消息 bit
            - 生成对应的“估计码字” c_hat

        返回:
            m_hat: 估计的消息比特 (0 或 1)
            c_hat: 对应的估计码字（长度 n）
        """
        y = np.array(y, dtype=int).flatten()

        if len(y) != self.n:
            raise ValueError(f"接收向量长度必须为 {self.n}，当前为 {len(y)}")

        ones = int(np.sum(y))
        zeros = self.n - ones

        # 多数表决：1 的数量多 -> 判为 1，反之为 0
        # 若刚好一样多（只有 n 为偶数才可能），这里简单地偏向 0
        m_hat = 1 if ones > zeros else 0
        c_hat = np.full(self.n, m_hat, dtype=int)
        return m_hat, c_hat


# ------------ 工具函数 & demo ------------

def create_repetition_code(n: int = 3) -> RepetitionCode:
    """
    工厂函数：创建一个长度为 n 的重复码。
    默认 n = 3，对应 (3,1,3) 码。
    """
    return RepetitionCode(n)


def run_demo():
    """
    直接运行本文件时的演示：
        - 指定 n 和消息 m
        - 编码 -> 加错误 -> 解码
        - 打印每一步的结果
    """
    n = 5
    m = 1  # 原始消息

    code = create_repetition_code(n)
    print(f"使用重复码 (n={n}, k=1, d={n})")
    print("原始消息 m:", m)

    # 编码
    c = code.encode(m)
    print("编码后码字 c:", c)

    # 在若干位置引入错误
    error_positions = [1, 3]  # 比如第 1 和第 3 号位置（从 0 开始计）
    y = code.add_errors(c, error_positions)
    print(f"在位置 {error_positions} 处引入翻转后的接收向量 y:", y)

    # 解码
    m_hat, c_hat = code.decode(y)
    print("解码得到的估计消息 m_hat:", m_hat)
    print("解码得到的估计码字 c_hat:", c_hat)

    print("消息是否恢复正确？", m_hat == m)
    print("码字是否恢复正确？", np.array_equal(c_hat, c))


if __name__ == "__main__":
    run_demo()
