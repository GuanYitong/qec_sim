# bit_phase_repetition/bit_repetition.py
from typing import List, Tuple


class BitRepetitionCode:
    """
    经典位重复码（bit-flip repetition code）。

    - 逻辑比特 m ∈ {0,1}
    - 编码：m -> [m, m, ..., m]（重复 n 次，n 必须为奇数）
    - 解码：对每 n 个物理比特做多数表决，可纠正最多 (n-1)/2 个比特翻转错误。

    使用方式：
        code = BitRepetitionCode(n=3)
        encoded = code.encode([1, 0, 1])
        decoded, flags = code.decode(encoded_with_errors)
    """

    def __init__(self, n: int = 3) -> None:
        if n <= 0 or n % 2 == 0:
            raise ValueError("n 必须是正奇数（例如 3, 5, 7...）。")
        self.n = n

    def encode_bit(self, m: int) -> List[int]:
        """
        编码单个逻辑比特。

        参数:
            m: 0 或 1

        返回:
            长度为 n 的物理比特列表。
        """
        if m not in (0, 1):
            raise ValueError("逻辑比特必须为 0 或 1。")
        return [m] * self.n

    def encode(self, bits: List[int]) -> List[int]:
        """
        编码比特序列。

        参数:
            bits: 由 0/1 组成的列表

        返回:
            物理比特列表，长度 = len(bits) * n
        """
        encoded: List[int] = []
        for b in bits:
            encoded.extend(self.encode_bit(b))
        return encoded

    def decode_block(self, block: List[int]) -> Tuple[int, bool]:
        """
        解码单个长度为 n 的 block（多数表决）。

        参数:
            block: 长度为 n 的 0/1 列表

        返回:
            (decoded_bit, error_detected)
            decoded_bit: 解码出的逻辑比特 0/1
            error_detected: 若 block 中存在与多数表决结果不一致的比特，则为 True
        """
        if len(block) != self.n:
            raise ValueError(f"block 长度必须为 {self.n}，当前为 {len(block)}。")
        if any(b not in (0, 1) for b in block):
            raise ValueError("block 中所有元素必须为 0 或 1。")

        ones = sum(block)
        zeros = self.n - ones
        decoded_bit = 1 if ones > zeros else 0
        corrected_block = [decoded_bit] * self.n
        error_detected = block != corrected_block
        return decoded_bit, error_detected

    def decode(self, received: List[int]) -> Tuple[List[int], List[bool]]:
        """
        解码整个码字序列。

        参数:
            received: 物理比特列表，长度必须是 n 的整数倍。

        返回:
            (decoded_bits, error_flags)
            decoded_bits: 解码得到的逻辑比特列表
            error_flags: 每个 block 是否检测到错误的布尔列表
        """
        if len(received) % self.n != 0:
            raise ValueError(
                f"received 长度必须是 {self.n} 的整数倍，当前为 {len(received)}。"
            )

        decoded_bits: List[int] = []
        error_flags: List[bool] = []

        for i in range(0, len(received), self.n):
            block = received[i : i + self.n]
            bit, flag = self.decode_block(block)
            decoded_bits.append(bit)
            error_flags.append(flag)

        return decoded_bits, error_flags

    def __repr__(self) -> str:
        return f"BitRepetitionCode(n={self.n})"
