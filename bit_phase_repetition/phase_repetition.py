# bit_phase_repetition/phase_repetition.py
from typing import List, Tuple

from .bit_repetition import BitRepetitionCode


class PhaseRepetitionCode(BitRepetitionCode):
    """
    “相位重复码”的经典模拟版本。

    在经典实现中依然使用 0/1 表示两个相位标签：
        - 0 可以理解为 “+ 相位”
        - 1 可以理解为 “- 相位”

    数学上与 BitRepetitionCode 完全相同，只是语义上表示
    “保护相位信息”，便于后续与量子相位翻转码对应。

    使用方式与 BitRepetitionCode 一致：
        code = PhaseRepetitionCode(n=3)
        encoded = code.encode([0, 1])
        decoded, flags = code.decode(encoded_with_phase_errors)
    """

    def __init__(self, n: int = 3) -> None:
        super().__init__(n=n)

    def __repr__(self) -> str:
        return f"PhaseRepetitionCode(n={self.n})"
