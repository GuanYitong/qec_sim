import sys, os
sys.path.append(os.path.abspath("."))   # 指向 qec_sim/
import numpy as np
from typing import List, Tuple

from classical.repetition import RepetitionCode


class Shor9BitFlipCode:
    """
    Shor(9,1,3) 码的一个“比特翻转错误玩具模型”：
    - 只在经典比特层面考虑 X 错误
    - 忽略相位（Z）和 Y 错误
    - 编码结构视为“双层重复码”：
        外层：3 比特重复码（逻辑 -> 3 个块）
        内层：每个块再用 3 比特重复码（块 -> 3 个物理比特）

    物理比特顺序约定为：
        block 0: qubits 0,1,2
        block 1: qubits 3,4,5
        block 2: qubits 6,7,8
    """

    def __init__(self):
        # 外层重复码：3 比特
        self.outer = RepetitionCode(n=3)
        # 内层重复码：每个块也是 3 比特
        self.inner = RepetitionCode(n=3)

        self.n = 9   # 物理比特数
        self.k = 1   # 逻辑比特数

    # ------------ 编码 ------------

    def encode_logical_bit(self, m: int) -> List[int]:
        """
        将单个逻辑比特 m ∈ {0,1} 编码为 9 个物理比特。

        步骤：
            1. 外层 repetition：m -> [b0, b1, b2]  (长度 3)
            2. 对每个 bi 进行内层 repetition：bi -> 3 比特
            3. 拼接得到长度 9 的码字
        """
        if m not in (0, 1):
            raise ValueError("逻辑比特 m 必须是 0 或 1")

        # 外层编码：得到 3 个“块比特”
        outer_bits = self.outer.encode(m)  # ndarray, shape = (3,)

        # 对每个块做内层编码
        phys_bits = []
        for b in outer_bits:
            inner_codeword = self.inner.encode(int(b))  # ndarray, shape = (3,)
            phys_bits.extend(inner_codeword.tolist())

        return phys_bits  # 长度 9 的 list[int]

    # ------------ 错误注入 ------------

    def add_bitflip_errors(self, codeword: List[int], error_positions: List[int]) -> List[int]:
        """
        在给定物理码字 codeword 的指定位置上施加比特翻转错误(X)。
        error_positions: 要翻转的物理位下标列表（0~8）。
        """
        y = codeword.copy()
        for pos in error_positions:
            if 0 <= pos < self.n:
                y[pos] ^= 1
        return y

    # ------------ 解码 ------------

    def decode(self, phys_bits: List[int]) -> Tuple[int, List[int], dict]:
        """
        对 9 个物理比特进行分层解码：
            1. 每个块 (3 比特) 使用内层 majority vote 解码成一个块比特
            2. 3 个块比特再用外层 majority vote 解码成逻辑比特

        返回:
            m_hat: 估计的逻辑比特 (0/1)
            block_bits: 每个块解码得到的 3 个块比特
            infos: 记录中间信息（例如各块中 0/1 计数）
        """
        if len(phys_bits) != self.n:
            raise ValueError(f"物理比特长度必须为 {self.n}")

        phys_bits = np.array(phys_bits, dtype=int) # type: ignore
        block_bits = []
        block_infos = []

        # 对每个块做内层 majority vote
        for block_idx in range(3):
            start = block_idx * 3
            end = start + 3
            block = phys_bits[start:end]

            ones = int(block.sum()) # type: ignore
            zeros = 3 - ones
            b_hat = 1 if ones > zeros else 0  # 平局时偏向 0

            block_bits.append(b_hat)
            block_infos.append({
                "block_idx": block_idx,
                "block_bits": block.tolist(), # type: ignore
                "zeros": zeros,
                "ones": ones,
                "decoded": b_hat
            })

        # 外层 majority vote：3 个块比特 -> 逻辑比特
        block_bits_arr = np.array(block_bits, dtype=int)
        ones_outer = int(block_bits_arr.sum())
        zeros_outer = 3 - ones_outer
        m_hat = 1 if ones_outer > zeros_outer else 0

        infos = {
            "block_infos": block_infos,
            "outer_zeros": zeros_outer,
            "outer_ones": ones_outer
        }

        return m_hat, block_bits, infos

    # ------------ 一个简单 demo ------------

    def demo_once(self, m: int, error_positions: List[int], verbose: bool = True):
        """
        对给定逻辑比特 m 和指定错误位置，演示一次编码-加错-解码流程。
        """
        codeword = self.encode_logical_bit(m)
        received = self.add_bitflip_errors(codeword, error_positions)
        m_hat, block_bits, infos = self.decode(received)

        if verbose:
            print("逻辑比特 m:", m)
            print("编码后物理码字:", codeword)
            print("注入 X 错误位置:", error_positions)
            print("接收向量:", received)
            print("每个块解码得到的比特:", block_bits)
            print("解码得到的逻辑比特 m_hat:", m_hat)
            print("是否纠正成功？", m_hat == m)

        return m, codeword, received, m_hat, infos


def run_demo():
    """
    直接运行本文件时的简单测试：
        - 给定一个逻辑比特
        - 随机在 9 个物理位中翻转 1 位
        - 看能否纠正
    """
    import random

    code = Shor9BitFlipCode()
    random.seed(0)

    for i in range(5):
        m = random.randint(0, 1)
        # 随机选一个错误位置
        pos = random.randint(0, 8)
        print(f"\n=== 实验 {i+1} ===")
        code.demo_once(m, [pos])


if __name__ == "__main__":
    run_demo()
