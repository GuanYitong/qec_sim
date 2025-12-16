# surface_d3.py

from typing import List, Dict, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class SurfaceCodeD3:
    """
    一个简化的 d=3 表面码模型，用于教学和仿真：
    - 9 个 data qubit，编号 0..8，布成 3x3 方阵
    - 4 个 Z 稳定子 ancilla，编号 9..12
    - 4 个 X 稳定子 ancilla，编号 13..16

    总共 17 个物理比特，拓扑上可以看成是一个 3x3 数据格点，
    每个 4-body 稳定子作用于一个小方块上的 4 个 data qubit。

    注意：这是一个“简化版 rotated surface code d=3”，主要用于示意完整纠错流程。
    """

    def __init__(self):
        # data qubits: 0..8
        self.data_qubits = list(range(9))
        # ancilla for Z-stabilizers: 9..12
        self.ancilla_z = list(range(9, 13))
        # ancilla for X-stabilizers: 13..16
        self.ancilla_x = list(range(13, 17))

        # 3x3 data qubits 的坐标 (row, col)，便于可视化或映射到物理芯片
        # row, col = 0,1,2
        self.data_coords: Dict[int, Tuple[int, int]] = {
            0: (0, 0), 1: (0, 1), 2: (0, 2),
            3: (1, 0), 4: (1, 1), 5: (1, 2),
            6: (2, 0), 7: (2, 1), 8: (2, 2),
        }

        # Z-type stabilizers 的邻接关系：
        # 这里每个 Z 稳定子作用在 4 个 data qubit 上：
        #  SZ0: D0,D1,D3,D4
        #  SZ1: D1,D2,D4,D5
        #  SZ2: D3,D4,D6,D7
        #  SZ3: D4,D5,D7,D8
        self.z_stabilizers = [
            {"name": "SZ0", "anc": self.ancilla_z[0], "data": [0, 1, 3, 4]},
            {"name": "SZ1", "anc": self.ancilla_z[1], "data": [1, 2, 4, 5]},
            {"name": "SZ2", "anc": self.ancilla_z[2], "data": [3, 4, 6, 7]},
            {"name": "SZ3", "anc": self.ancilla_z[3], "data": [4, 5, 7, 8]},
        ]

        # X-type stabilizers 的邻接关系：
        # 为了简化教学，我们让它和 Z stabilizer 拓扑相同
        # （在真正的 surface code 中，X/Z 稳定子是棋盘格交错的，这里只强调流程）
        self.x_stabilizers = [
            {"name": "SX0", "anc": self.ancilla_x[0], "data": [0, 1, 3, 4]},
            {"name": "SX1", "anc": self.ancilla_x[1], "data": [1, 2, 4, 5]},
            {"name": "SX2", "anc": self.ancilla_x[2], "data": [3, 4, 6, 7]},
            {"name": "SX3", "anc": self.ancilla_x[3], "data": [4, 5, 7, 8]},
        ]

        # 总 qubit 数
        self.n_qubits = 17

        # 初始化一个简单的单轮解码 lookup 表
        self._init_single_error_decoder()

    # ------------------------------------------------------------------
    # 拓扑相关工具
    # ------------------------------------------------------------------
    def print_lattice_info(self):
        print("Surface code d=3 (simplified) lattice:")
        print("Data qubits (3x3 grid):")
        for q, (r, c) in self.data_coords.items():
            print(f"  D{q}: row={r}, col={c}")
        print("\nZ stabilizers:")
        for i, stab in enumerate(self.z_stabilizers):
            print(f"  {stab['name']} (ancilla q{stab['anc']}), data = {stab['data']}")
        print("\nX stabilizers:")
        for i, stab in enumerate(self.x_stabilizers):
            print(f"  {stab['name']} (ancilla q{stab['anc']}), data = {stab['data']}")

    # ------------------------------------------------------------------
    # 一轮 syndrome 提取电路
    # ------------------------------------------------------------------
    def build_one_round_circuit(self) -> QuantumCircuit:
        """
        构造一轮 syndrome 提取的量子线路：
        - 对每个 Z 稳定子：
            ancilla 准备 |0>（默认即为 |0>）
            用 data 为控制，ancilla 为靶的 CNOT 收集 Z-parity
            测量 ancilla 到经典比特 c[0..3]
        - 对每个 X 稳定子：
            ancilla 准备 |+>：H |0>
            用 ancilla 为控制，data 为靶做 CNOT
            再对 ancilla 做 H，相当于在 X 基测量
            测量 ancilla 到经典比特 c[4..7]
        """
        qreg = QuantumRegister(self.n_qubits, "q")
        creg = ClassicalRegister(8, "c")  # 0..3: Z syndromes; 4..7: X syndromes
        qc = QuantumCircuit(qreg, creg, name="SurfaceCode_d3_round")

        # --- Z stabilizer measurement ---
        for i, stab in enumerate(self.z_stabilizers):
            anc = stab["anc"]
            # 假定 anc 初始为 |0>，如需反复轮次可以在此加 reset(qreg[anc])
            for dq in stab["data"]:
                # data 为 control，ancilla 为 target：收集 Z-parity
                qc.cx(qreg[dq], qreg[anc])
            qc.measure(qreg[anc], creg[i])

        # --- X stabilizer measurement ---
        for j, stab in enumerate(self.x_stabilizers):
            anc = stab["anc"]
            cidx = 4 + j
            # 准备 |+> = H|0>
            qc.h(qreg[anc])
            # ancilla 为 control，data 为 target，收集 X-parity
            for dq in stab["data"]:
                qc.cx(qreg[anc], qreg[dq])
            # 再做一次 H，相当于在 X 基测量
            qc.h(qreg[anc])
            qc.measure(qreg[anc], creg[cidx])

        return qc

    # ------------------------------------------------------------------
    # 非实时：基于综合征的单轮解码（只处理单比特错误）
    # ------------------------------------------------------------------
    def _init_single_error_decoder(self):
        """
        为 d=3 的简化 surface code 构造一个“单比特 X/Z 错误”的解码 lookup 表。
        这里不实现 MWPM，而是针对单个错误构造综合征 → data qubit 的映射，
        用于教学和 demo。
        """
        # 对于 X 错误：会与 Z stabilizer 反对易，所以看 Z 综合征 (前 4 bit)
        self.lookup_Z_syndrome_to_data: Dict[Tuple[int, int, int, int], int] = {}
        for dq in self.data_qubits:
            # dq 上发生 X 错误，对所有 Z stabilizer 看它是否包含 dq
            pattern = []
            for stab in self.z_stabilizers:
                pattern.append(1 if dq in stab["data"] else 0)
            self.lookup_Z_syndrome_to_data[tuple(pattern)] = dq

        # 对于 Z 错误：会与 X stabilizer 反对易，所以看 X 综合征 (后 4 bit)
        self.lookup_X_syndrome_to_data: Dict[Tuple[int, int, int, int], int] = {}
        for dq in self.data_qubits:
            pattern = []
            for stab in self.x_stabilizers:
                pattern.append(1 if dq in stab["data"] else 0)
            self.lookup_X_syndrome_to_data[tuple(pattern)] = dq

    def decode_single_round(self, syndrome: List[int]) -> Dict[str, List[int]]:
        """
        基于 8 比特综合征（[sZ0,sZ1,sZ2,sZ3,sX0,sX1,sX2,sX3]）
        给出一个“单比特错误假设”的纠正方案：
            - 对 sZ (前 4 bit)，找到对应的 data qubit，施加 X 纠正
            - 对 sX (后 4 bit)，找到对应的 data qubit，施加 Z 纠正
        返回:
            {
                "X": [需要施加 X 的 data qubit 列表],
                "Z": [需要施加 Z 的 data qubit 列表],
            }
        若综合征不在 lookup 表中（例如多比特错误），则对应列表为空。
        """
        if len(syndrome) != 8:
            raise ValueError("syndrome 长度必须为 8")

        sZ = tuple(syndrome[0:4])
        sX = tuple(syndrome[4:8])

        corrections_X: List[int] = []
        corrections_Z: List[int] = []

        if sZ in self.lookup_Z_syndrome_to_data and any(sZ):
            corrections_X.append(self.lookup_Z_syndrome_to_data[sZ])

        if sX in self.lookup_X_syndrome_to_data and any(sX):
            corrections_Z.append(self.lookup_X_syndrome_to_data[sX])

        return {"X": corrections_X, "Z": corrections_Z}

    # ------------------------------------------------------------------
    # 一个演示用的高层函数：注入错误 → 测量综合征 → 解码
    # ------------------------------------------------------------------
    def build_round_with_errors(
        self,
        x_errors: Optional[List[int]] = None,
        z_errors: Optional[List[int]] = None,
    ) -> QuantumCircuit:
        """
        构造一轮：在指定 data qubits 上注入 X/Z 错误，然后进行一次 syndrome 提取。

        参数:
            x_errors: 需要施加 X 错误的 data qubit 列表
            z_errors: 需要施加 Z 错误的 data qubit 列表
        """
        qreg = QuantumRegister(self.n_qubits, "q")
        creg = ClassicalRegister(8, "c")
        qc = QuantumCircuit(qreg, creg, name="SurfaceCode_d3_with_errors")

        x_errors = x_errors or []
        z_errors = z_errors or []

        # 1) 注入 X/Z 错误（这里直接在 data qubits 上加 Pauli X/Z）
        for dq in x_errors:
            qc.x(qreg[dq])
        for dq in z_errors:
            qc.z(qreg[dq])

        # 2) 拼接一轮 syndrome 提取
        round_circ = self.build_one_round_circuit()
        qc.compose(round_circ, qreg[:] + creg[:], inplace=True)

        return qc
