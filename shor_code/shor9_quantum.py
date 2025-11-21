"""
shor9_quantum.py

Shor 9-qubit code 的量子版示例实现（Qiskit）：
- 构造 9 比特码的编码线路（|0_L> 或 |1_L>）
- 支持在某个物理比特上施加单比特 Pauli 错误
- 用“编码单元的逆电路”作为解码器，演示单比特错误被纠正的效果
  （在理想模拟器中，反向编码等价于一个“完美解码+纠错”的过程）

注意：
- 这是教学用玩具实现，未包含稳定子测量、fault-tolerant 设计等工程细节。
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


class Shor9QuantumCode:
    """Shor(9,1,3) 量子版编码/解码简单模型。"""

    def __init__(self):
        self.n_phys = 9  # 物理比特数
        self.n_logical = 1

    # ------------------------------------------------------------------
    # 1. 编码电路
    # ------------------------------------------------------------------
    def build_encode_circuit(self, logical_state: str = "0") -> QuantumCircuit:
        """
        构造一个 9 qubit 的编码电路，将 |0> 或 |1> 编码成 Shor 码。

        logical_state: "0" 或 "1"，表示逻辑态 |0_L> 或 |1_L>。
        返回：只包含编码过程的 QuantumCircuit（不含测量）。
        """
        qc = QuantumCircuit(self.n_phys, name="ShorEncode")

        # 逻辑信息初始放在 qubit 0 上
        if logical_state == "1":
            qc.x(0)

        # 第一步：相位编码（生成 GHZ 态）
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)

        # 第二步：对每个“相位块”做 3 比特重复编码
        # 块1: qubits 0,3,6
        qc.cx(0, 3)
        qc.cx(0, 6)

        # 块2: qubits 1,4,7
        qc.cx(1, 4)
        qc.cx(1, 7)

        # 块3: qubits 2,5,8
        qc.cx(2, 5)
        qc.cx(2, 8)

        return qc

    # ------------------------------------------------------------------
    # 2. 在某个物理比特上施加 Pauli 错误
    # ------------------------------------------------------------------
    @staticmethod
    def apply_pauli_error(
        qc: QuantumCircuit, qubit: int, pauli: str = "X"
    ) -> None:
        """
        在给定电路 qc 的某个物理比特 qubit 上附加一个 Pauli 错误门。

        pauli: "X", "Y", 或 "Z"
        直接在传入的电路上就地修改，不返回新电路。
        """
        pauli = pauli.upper()
        if pauli == "X":
            qc.x(qubit)
        elif pauli == "Y":
            qc.y(qubit)
        elif pauli == "Z":
            qc.z(qubit)
        else:
            raise ValueError("pauli 必须为 'X', 'Y' 或 'Z'")

    # ------------------------------------------------------------------
    # 3. 构造“编码 → 施加错误 → 反向解码 → 测量”的 demo 电路
    # ------------------------------------------------------------------
    def build_encode_error_decode_circuit(
        self,
        logical_state: str = "0",
        error_qubit: int | None = None,
        error_pauli: str = "X",
    ) -> QuantumCircuit:
        """
        构造一个完整流程的电路：
            1) 编码逻辑态 |0_L> 或 |1_L>
            2) 可选：在某个物理比特上施加一个 Pauli 错误
            3) 应用编码电路的逆（作为解码过程）
            4) 测量第 0 个物理比特（对应逻辑比特）

        在理想无噪声模拟下：
            - 若错误可被 Shor 码纠正，则测量结果应等于初始 logical_state。
        """
        # 注册
        qreg = QuantumRegister(self.n_phys, "q")
        creg = ClassicalRegister(1, "c")

        qc = QuantumCircuit(qreg, creg, name="ShorEncodeErrorDecode")

        # 编码
        encode_circ = self.build_encode_circuit(logical_state)
        qc.compose(encode_circ, inplace=True)

        # 错误
        if error_qubit is not None:
            self.apply_pauli_error(qc, error_qubit, error_pauli)

        # 解码
        decode_circ = encode_circ.inverse()
        qc.compose(decode_circ, inplace=True)

        # 测量逻辑比特（物理比特0）
        qc.measure(qreg[0], creg[0])

        return qc

    # ------------------------------------------------------------------
    # 4. 简单命令行 demo：需要用户具备 Qiskit 运行环境
    # ------------------------------------------------------------------
    def run_demo(
        self,
        logical_state: str = "0",
        error_qubit: int | None = 0,
        error_pauli: str = "X",
        shots: int = 1024,
    ):
        """
        使用 Qiskit 的 Aer 或 BasicSimulator 运行一个简单 demo：
            - 编码
            - 施加单比特错误
            - 反向解码
            - 测量逻辑比特
        输出测量统计结果。
        """
        from qiskit_aer import Aer  # 推荐使用 Aer
        backend = Aer.get_backend("aer_simulator")

        qc = self.build_encode_error_decode_circuit(
            logical_state=logical_state,
            error_qubit=error_qubit,
            error_pauli=error_pauli,
        )

        # 对于 Aer，需要先转换为测量兼容电路
        qc = qc.copy()
        qc = qc.decompose()  # 展开复合门便于查看

        from qiskit import transpile

        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result() #type: ignore
        counts = result.get_counts()

        print("逻辑初态:", logical_state)
        print("错误位置:", error_qubit, "错误类型:", error_pauli)
        print("测量结果统计:", counts)
        return counts


# 如果直接运行本文件，给一个简单命令行 demo
if __name__ == "__main__":
    code = Shor9QuantumCode()
    # 示例：逻辑 |0_L>，在物理比特 2 上施加 X 错误
    code.run_demo(logical_state="0", error_qubit=2, error_pauli="X")
