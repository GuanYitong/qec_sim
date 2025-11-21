import sys, os
sys.path.append(os.path.abspath("."))   # 指向 qec_sim/
import numpy as np
from typing import List, Tuple

from classical.hamming74 import LinearBlockCode, create_hamming_74


class SteaneCode:
    """
    经典稳定子仿真模型的 Steane [[7,1,3]] 码。
    只模拟 X / Z Pauli 错误的 syndrome 和纠正。
    """

    def __init__(self):
        # 使用 Hamming(7,4) 构造 CSS 稳定子
        hamming = create_hamming_74()
        self.H = hamming.H        # 3 x 7

        self.n = 7   # 物理比特
        self.k = 1   # 逻辑比特

    # -----------------------------------------------------------
    # 1) 编码：只编码 |0> 和 |1>（经典化，返回 7 比特表示）
    # -----------------------------------------------------------
    def encode_logical_bit(self, m: int) -> List[int]:
        """
        教学模型：用简单经典方式表示逻辑态
        |0_L> ~ 0000000
        |1_L> ~ 1111111
        实际 Steane 码是叠加态，但这个经典模型用于展示 syndrome 和纠错流程。
        """
        if m not in (0,1):
            raise ValueError("逻辑比特必须为 0 或 1")
        return [m] * 7

    # -----------------------------------------------------------
    # 2) 注入错误（Pauli X / Z）
    #   使用两个长度为 7 的向量：
    #      bits_X[i] = 1 表示第 i 位出现 X 错误
    #      bits_Z[i] = 1 表示第 i 位出现 Z 错误
    # -----------------------------------------------------------
    def apply_errors(self, codeword: List[int],
                     X_errors: List[int], Z_errors: List[int]):
        """
        在 codeword 上施加 X / Z 错误。
        返回：
            received_bits: bit 翻转后的结果
            X_mask, Z_mask: 记录哪些位置犯了 X 或 Z 错误
        """
        cw = np.array(codeword, dtype=int)
        X_mask = np.array(X_errors, dtype=int)
        Z_mask = np.array(Z_errors, dtype=int)

        # X 错误等价于翻转比特：0->1, 1->0
        received = (cw ^ X_mask) % 2

        return received.tolist(), X_mask.tolist(), Z_mask.tolist()

    # -----------------------------------------------------------
    # 3) syndrome 计算
    #   Z 错误由 X 稳定子检测： syndrome_X = Z_mask * H^T
    #   X 错误由 Z 稳定子检测： syndrome_Z = X_mask * H^T
    # -----------------------------------------------------------
    def syndrome(self, X_mask: List[int], Z_mask: List[int]):
        X_mask = np.array(X_mask, dtype=int) # type: ignore
        Z_mask = np.array(Z_mask, dtype=int) # type: ignore

        # X 错误 → Z 稳定子检测
        syn_Z = (X_mask @ self.H.T) % 2
        # Z 错误 → X 稳定子检测
        syn_X = (Z_mask @ self.H.T) % 2

        return syn_X.tolist(), syn_Z.tolist()

    # -----------------------------------------------------------
    # 4) 解码：根据 Hamming 的 syndrome 查表纠错
    # -----------------------------------------------------------
    def decode(self, received: List[int], X_mask: List[int], Z_mask: List[int]):
        """
        输入：
            received: 7 比特（已被 X 翻转影响）
            X_mask, Z_mask: 记录的错误位置（用于生成 syndrome）
        输出：
            修正后的结果、估计错误向量
        """
        # 计算 syndrome
        syn_X, syn_Z = self.syndrome(X_mask, Z_mask)

        # 使用 Hamming syndrome 查表法
        hamming = create_hamming_74()

        # 纠正 X 错误：即针对 Z-mask
        eZ = hamming.syndrome_table.get(tuple(syn_X), np.zeros(7, dtype=int))
        # 纠正 Z 错误：即针对 X-mask
        eX = hamming.syndrome_table.get(tuple(syn_Z), np.zeros(7, dtype=int))

        # 修正后的比特值
        r = np.array(received, dtype=int)
        corrected = (r ^ eX) % 2  # X错误导致比特翻转

        return corrected.tolist(), eX.tolist(), eZ.tolist(), syn_X, syn_Z

    # -----------------------------------------------------------
    # 简单 demo
    # -----------------------------------------------------------
    def demo_once(self, m: int, Xerr: List[int], Zerr: List[int]):
        codeword = self.encode_logical_bit(m)
        received, Xm, Zm = self.apply_errors(codeword, Xerr, Zerr)
        corrected, eX, eZ, synX, synZ = self.decode(received, Xm, Zm)

        print("逻辑比特:", m)
        print("编码码字:", codeword)
        print("施加 X 错误:", Xerr)
        print("施加 Z 错误:", Zerr)
        print("收到比特:", received)
        print("综合征 X:", synX, "   综合征 Z:", synZ)
        print("估计 X 错误 eX:", eX)
        print("估计 Z 错误 eZ:", eZ)
        print("纠正后比特:", corrected)
        return corrected
