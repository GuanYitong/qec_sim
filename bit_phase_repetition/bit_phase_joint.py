# bit_phase_repetition/bit_phase_joint.py
from typing import List, Tuple, Dict

from .bit_repetition import BitRepetitionCode
from .phase_repetition import PhaseRepetitionCode


class BitPhaseJointCode:
    """
    位 + 相位 串联重复码（经典 3×3 Shor 结构的简化模拟）。

    结构：
        - 外层：PhaseRepetitionCode，重复 n_phase 次（相位方向）
        - 内层：BitRepetitionCode，重复 n_bit 次（位翻转方向）

    对单个逻辑比特 m 的编码：
        1) 相位重复：得到 [m, m, ..., m]  (长度 n_phase)
        2) 对每个相位比特再做位重复：每个 m -> [m,...,m] (长度 n_bit)
        3) 拼接得到长度 n_phase * n_bit 的码字

    因此：
        对 k 个逻辑比特，总物理比特数为 k * n_phase * n_bit

    解码流程（对每个逻辑比特）：
        1) 按相位方向分成 n_phase 个 block，每个 block 长度 n_bit
        2) 对每个 block 用 BitRepetitionCode 多数表决，纠正 bit-flip，得到 n_phase 个“相位比特”
        3) 对这 n_phase 个“相位比特”再用 PhaseRepetitionCode 多数表决，纠正 phase 类型的差异
        4) 最终得到一个逻辑比特

    这里的“phase error” 仍然是用 0/1 标签来模拟，真正的物理含义会在量子版本中体现。
    """

    def __init__(self, n_bit: int = 3, n_phase: int = 3) -> None:
        if n_bit <= 0 or n_bit % 2 == 0:
            raise ValueError("n_bit 必须是正奇数（例如 3, 5, 7...）。")
        if n_phase <= 0 or n_phase % 2 == 0:
            raise ValueError("n_phase 必须是正奇数（例如 3, 5, 7...）。")

        self.n_bit = n_bit
        self.n_phase = n_phase

        self.bit_code = BitRepetitionCode(n_bit)
        self.phase_code = PhaseRepetitionCode(n_phase)

    # ---------- 编码部分 ----------

    def encode_bit(self, m: int) -> List[int]:
        """
        编码单个逻辑比特（先相位重复，再位重复）。

        参数:
            m: 0 或 1

        返回:
            长度为 n_phase * n_bit 的物理比特列表。
        """
        if m not in (0, 1):
            raise ValueError("逻辑比特必须为 0 或 1。")

        encoded_blocks: List[int] = []

        # 外层：相位重复 n_phase 次（只是逻辑上的标签复制）
        phase_bits = self.phase_code.encode([m])  # 结果是 [m, m, ..., m]（长度 n_phase）

        # 内层：对每个“相位比特”再做位重复
        for phase_bit in phase_bits:
            encoded_blocks.extend(self.bit_code.encode_bit(phase_bit))

        return encoded_blocks

    def encode(self, bits: List[int]) -> List[int]:
        """
        编码比特序列。

        参数:
            bits: 逻辑比特列表

        返回:
            物理比特列表，长度 = len(bits) * n_phase * n_bit
        """
        encoded: List[int] = []
        for b in bits:
            encoded.extend(self.encode_bit(b))
        return encoded

    # ---------- 解码部分 ----------

    def _decode_single_logical(
        self, block: List[int]
    ) -> Tuple[int, Dict[str, object]]:
        """
        解码单个逻辑比特对应的完整物理 block。

        参数:
            block: 长度为 n_phase * n_bit 的列表

        返回:
            (decoded_bit, info)
            info 包含：
                - "inner_bit_error_flags": List[bool]，长度 n_phase，每个值表示对应相位块的 bit-flip 是否被纠正
                - "phase_error_detected": bool，相位层是否检测到不一致
                - "intermediate_phase_bits": List[int]，内层解码得到的 n_phase 个“相位比特”
        """
        if len(block) != self.n_phase * self.n_bit:
            raise ValueError(
                f"单个逻辑比特的 block 长度必须为 {self.n_phase * self.n_bit}，当前为 {len(block)}。"
            )

        # 1) 先按相位方向切分成 n_phase 个 block，每块长度 n_bit
        phase_blocks: List[List[int]] = []
        for i in range(self.n_phase):
            start = i * self.n_bit
            end = start + self.n_bit
            phase_blocks.append(block[start:end])

        # 2) 对每个 block 用 BitRepetitionCode 解码（纠正 bit-flip）
        intermediate_phase_bits: List[int] = []
        inner_bit_error_flags: List[bool] = []

        for pb in phase_blocks:
            bit, flag = self.bit_code.decode_block(pb)
            intermediate_phase_bits.append(bit)
            inner_bit_error_flags.append(flag)

        # 3) 对 n_phase 个“相位比特”再做 PhaseRepetitionCode 多数表决
        logical_bit, phase_error_flags = self.phase_code.decode(intermediate_phase_bits)
        # phase_error_flags 是长度 1 的列表
        phase_error_detected = phase_error_flags[0]

        info = {
            "inner_bit_error_flags": inner_bit_error_flags,
            "phase_error_detected": phase_error_detected,
            "intermediate_phase_bits": intermediate_phase_bits,
        }

        return logical_bit, info # type: ignore

    def decode(self, received: List[int]) -> Tuple[List[int], List[Dict[str, object]]]:
        """
        解码整个物理比特序列。

        参数:
            received: 物理比特列表，长度必须是 n_phase * n_bit 的整数倍。

        返回:
            (decoded_bits, infos)
            decoded_bits: 逻辑比特列表
            infos: 每个逻辑比特对应的详细信息字典列表
        """
        block_len = self.n_phase * self.n_bit
        if len(received) % block_len != 0:
            raise ValueError(
                f"received 长度必须是 {block_len} 的整数倍，当前为 {len(received)}。"
            )

        num_logical = len(received) // block_len
        decoded_bits: List[int] = []
        infos: List[Dict[str, object]] = []

        for i in range(num_logical):
            start = i * block_len
            end = start + block_len
            block = received[start:end]
            bit, info = self._decode_single_logical(block)
            decoded_bits.append(bit)
            infos.append(info)

        return decoded_bits, infos

    def __repr__(self) -> str:
        return f"BitPhaseJointCode(n_bit={self.n_bit}, n_phase={self.n_phase})"
