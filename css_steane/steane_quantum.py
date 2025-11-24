"""
steane_quantum.py

Steane [[7,1,3]] 码的量子稳定子测量电路（Qiskit 实现）。

作用：
- 定义 Steane 码的 6 个稳定子生成元（3 个 X 型，3 个 Z 型）
- 给定 7 个物理比特上的状态（理论上应是 Steane 编码态），
  构造一个带 6 个 ancilla 的量子电路，测量所有稳定子，得到 6 比特综合征。

注意：
- 这里只实现“量子测量稳定子”的部分；
- 编码/解码（把 1 个逻辑 qubit 编到 7 个物理 qubit 中）目前由经典 steane_code 负责逻辑，
  之后可以再补真正的量子编码电路。
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Optional


class SteaneQuantumCode:
    """
    Steane [[7,1,3]] 码的量子稳定子测量模块。
    """

    def __init__(self):
        self.n_phys = 7     # 物理比特数
        self.n_logical = 1  # 逻辑比特数

        # Steane 码的标准稳定子生成元（长度 7 的 Pauli 串）
        # 约定 qubits 顺序为 q0, q1, ..., q6
        #
        # X 型稳定子：
        #   g1^X = X X X X I I I
        #   g2^X = X X I I X X I
        #   g3^X = X I X I X I X
        #
        # Z 型稳定子：
        #   g1^Z = Z Z Z Z I I I
        #   g2^Z = Z Z I I Z Z I
        #   g3^Z = Z I Z I Z I Z

        self.X_generators = [
            "XXXXIII",
            "XXIIXXI",
            "XIXIXIX",
        ]
        self.Z_generators = [
            "ZZZZIII",
            "ZZIIZZI",
            "ZIZIZIZ",
        ]

    # ------------------------------------------------------------------
    # 1) 内部工具：测量单个 Pauli 稳定子 S (例如 "XIXIZII")
    # ------------------------------------------------------------------

    def _measure_pauli_string(
        self,
        qc: QuantumCircuit,
        data: QuantumRegister,
        ancilla,
        clbit,
        pauli: str,
    ) -> None:
        """
        使用一个 ancilla qubit 测量给定 Pauli 串的本征值（±1）：
            - pauli: 长度为 7 的字符串，只包含 I/X/Z (Steane 码无 Y)

        协议（标准做法）：
            1. 对所有带 X 的 data qubit 先做 H，使 X -> Z
            2. 用 data 作为 control，ancilla 作为 target 做 CNOT，收集 Z-parity
            3. 再对带 X 的 data qubit 做一次 H 还原
            4. 测量 ancilla 的 Z 基，得到 0/1（0 表示 +1 本征值，1 表示 -1 本征值）
        """
        if len(pauli) != len(data):
            raise ValueError("Pauli 串长度必须等于数据比特数")

        # 1) X -> Z 变换
        x_positions = []
        for i, p in enumerate(pauli):
            if p == "X":
                x_positions.append(i)
                qc.h(data[i])

        # 2) 对所有非 I 的位置做 CNOT(data -> ancilla)
        for i, p in enumerate(pauli):
            if p in ("X", "Z"):  # 已经把 X 映射到 Z 了
                qc.cx(data[i], ancilla)

        # 3) X 位置再做一次 H 还原
        for i in x_positions:
            qc.h(data[i])

        # 4) 测量 ancilla
        qc.measure(ancilla, clbit)

    # ------------------------------------------------------------------
    # 2) 构造完整的稳定子测量电路
    # ------------------------------------------------------------------

    def build_syndrome_measure_circuit(
        self,
        add_error: Optional[tuple] = None,
    ) -> QuantumCircuit:
        """
        构造用于 Steane 码稳定子测量的量子电路。

        比特布局：
            - data: 7 个物理数据比特，命名为 q[0..6]
            - anc:  6 个辅助比特，分别用于 3 个 Z 稳定子和 3 个 X 稳定子
            - c:    6 个经典比特，记录测量结果
              约定：
                c[0..2] -> Z 型稳定子综合征
                c[3..5] -> X 型稳定子综合征

        参数:
            add_error: 可选 (qubit_index, pauli_str)，例如 (2, "X")
                       表示在数据比特 data[2] 上加一个 Pauli X 错误，仅用于教学演示。

        返回:
            qc: 构造好的 QuantumCircuit
        """
        data = QuantumRegister(self.n_phys, "q")
        anc = QuantumRegister(6, "anc")
        creg = ClassicalRegister(6, "c")

        qc = QuantumCircuit(data, anc, creg, name="SteaneSyndrome")

        # （可选）添加一个单比特 Pauli 错误，用于演示综合征变化
        if add_error is not None:
            qubit_idx, pauli = add_error
            pauli = pauli.upper()
            if pauli == "X":
                qc.x(data[qubit_idx])
            elif pauli == "Z":
                qc.z(data[qubit_idx])
            elif pauli == "Y":
                qc.y(data[qubit_idx])
            else:
                raise ValueError("Pauli 错误类型必须为 'X','Y','Z' 中之一")

        # 先测量 3 个 Z 型稳定子：使用 anc[0..2], c[0..2]
        for i, gen in enumerate(self.Z_generators):
            self._measure_pauli_string(
                qc,
                data=data,
                ancilla=anc[i],
                clbit=creg[i],
                pauli=gen,
            )

        # 再测量 3 个 X 型稳定子：使用 anc[3..5], c[3..5]
        for i, gen in enumerate(self.X_generators):
            self._measure_pauli_string(
                qc,
                data=data,
                ancilla=anc[3 + i],
                clbit=creg[3 + i],
                pauli=gen,
            )

        return qc

    # ------------------------------------------------------------------
    # 3) 一个简单 demo：在命令行下跑
    # ------------------------------------------------------------------

    def run_demo(self):
        """
        简单 demo：
            - 在 q2 上加一个 X 错误
            - 测量所有稳定子
            - 打印综合征统计

        注意：这里没有真正做“编码”，只是演示稳定子测量电路本身的使用方法。
        真正使用时，应先准备好 Steane 编码态，再串接本电路。
        """
        from qiskit_aer import Aer
        from qiskit import transpile

        backend = Aer.get_backend("aer_simulator")

        qc = self.build_syndrome_measure_circuit(add_error=(2, "X"))
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=1024).result() #type: ignore
        counts = result.get_counts()

        print("在 q2 上施加 X 错误后的综合征测量结果统计：")
        print(counts)
        return counts


if __name__ == "__main__":
    code = SteaneQuantumCode()
    code.run_demo()
