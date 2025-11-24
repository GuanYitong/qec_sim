# stabilizer_code.py

from typing import List, Tuple


def pauli_anticommute(p1: str, p2: str) -> bool:
    """
    判断两个单比特 Pauli 算符是否反对易：
        X, Y, Z 之间两两反对易，其它情况（含 I）都对易。
    """
    if p1 == "I" or p2 == "I":
        return False
    if p1 == p2:
        return False
    # X,Y,Z 任意不同一对都反对易
    return True


def pauli_string_anticommute(s1: str, s2: str) -> bool:
    """
    判断两个 n 比特 Pauli 串是否整体反对易：
        在每个比特上检查局域 Pauli 是否反对易，
        若反对易位置数为奇数，则整体反对易。
    """
    if len(s1) != len(s2):
        raise ValueError("两个 Pauli 串长度必须一致")
    count = 0
    for a, b in zip(s1, s2):
        if pauli_anticommute(a, b):
            count += 1
    # 奇数次反对易 => 整体反对易
    return (count % 2) == 1


class StabilizerCode:
    """
    通用稳定子码框架（不演化量子态，只做“错误 -> 综合征”这一层）。

    用法：
        - n: 物理比特数
        - stabilizers: 稳定子生成元列表，每个是长度为 n 的 Pauli 串，如 "XXI", "ZZI..."
          字符只支持 "I","X","Y","Z"。
    """

    def __init__(self, n: int, stabilizers: List[str]):
        self.n = n
        # 规范化 Pauli 串
        self.stabilizers = [s.upper() for s in stabilizers]
        for s in self.stabilizers:
            if len(s) != n:
                raise ValueError("稳定子长度必须等于 n")
            for ch in s:
                if ch not in "IXYZ":
                    raise ValueError("稳定子中只能包含 I,X,Y,Z")

    def syndrome(self, error: str) -> List[int]:
        """
        计算给定错误 Pauli 串的综合征：
            对每个稳定子 g_i，若 error 与 g_i 反对易，则综合征位为 1，否则为 0。

        参数:
            error: 长度为 n 的 Pauli 串，如 "XII...Z"
        返回:
            syndrome: 长度为 len(stabilizers) 的 0/1 列表
        """
        error = error.upper()
        if len(error) != self.n:
            raise ValueError("错误 Pauli 串长度必须等于 n")
        for ch in error:
            if ch not in "IXYZ":
                raise ValueError("错误 Pauli 串中只能包含 I,X,Y,Z")

        syn = []
        for g in self.stabilizers:
            syn.append(1 if pauli_string_anticommute(g, error) else 0)
        return syn

    def print_info(self):
        print(f"Stabilizer code with n = {self.n}")
        print("Stabilizer generators:")
        for i, g in enumerate(self.stabilizers):
            print(f"  g{i+1} = {g}")


# ------ 一个简单示例：3-qubit bit-flip 码的 X/Z 稳定子 ------
def create_3qubit_bitflip_stabilizer() -> StabilizerCode:
    """
    3-qubit bit-flip code 的典型稳定子表示：
        g1 = Z Z I
        g2 = I Z Z

    用来检测 X 错误（比特翻转），这里只给出最简单示例。
    """
    n = 3
    stabilizers = [
        "ZZI",
        "IZZ",
    ]
    return StabilizerCode(n, stabilizers)


if __name__ == "__main__":
    # 小测试：在 3-qubit bit-flip 码上测试几个错误的综合征
    code = create_3qubit_bitflip_stabilizer()
    code.print_info()

    errors = ["XII", "IXI", "IIX", "XXX", "ZII", "IIZ"]
    for e in errors:
        syn = code.syndrome(e)
        print(f"error {e} -> syndrome {syn}")
