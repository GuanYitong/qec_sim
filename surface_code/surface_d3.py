# surface_code/surface_d3.py
# Teaching implementation of rotated d=3 surface code (Surface-17 patch)
# - 9 data qubits: d0..d8
# - 4 Z-plaquette ancilla: az0..az3 (weight-4)
# - 4 X-boundary ancilla: ax0..ax3 (weight-2 in this teaching patch)
#
# This module focuses on:
# - clear topology definition
# - modular stabilizer-measurement circuits (Z-check / X-check)
# - a full round circuit builder with barriers
# - error injection + syndrome parsing helpers
# - a toy single-error decoder for demonstrations
#
# Note: For production-grade decoding, replace toy lookup with multi-round + MWPM.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import re
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


@dataclass(frozen=True)
class StabilizerSpec:
    name: str
    anc: str
    data: Tuple[str, ...]


class Surface17D3:
    """Rotated d=3 surface code (Surface-17 teaching patch)."""

    def __init__(self):
        self.data = [f"d{i}" for i in range(9)]
        self.az   = [f"az{i}" for i in range(4)]
        self.ax   = [f"ax{i}" for i in range(4)]

        self.data_coords: Dict[str, Tuple[float, float]] = {
            "d0": (0,2), "d1": (1,2), "d2": (2,2),
            "d3": (0,1), "d4": (1,1), "d5": (2,1),
            "d6": (0,0), "d7": (1,0), "d8": (2,0),
        }

        self.z_checks: List[StabilizerSpec] = [
            StabilizerSpec("Zp0","az0",("d0","d1","d3","d4")),
            StabilizerSpec("Zp1","az1",("d1","d2","d4","d5")),
            StabilizerSpec("Zp2","az2",("d3","d4","d6","d7")),
            StabilizerSpec("Zp3","az3",("d4","d5","d7","d8")),
        ]

        self.x_checks: List[StabilizerSpec] = [
            StabilizerSpec("Xs0","ax0",("d0","d1")),  # top
            StabilizerSpec("Xs1","ax1",("d2","d5")),  # right
            StabilizerSpec("Xs2","ax2",("d3","d6")),  # left
            StabilizerSpec("Xs3","ax3",("d7","d8")),  # bottom
        ]

        # toy single-error lookup
        self._x_lookup = self._build_single_X_lookup()
        self._z_lookup = self._build_single_Z_lookup()

    # ------------------------------------------------------------------
    # Named registers (readable circuit diagrams)
    # ------------------------------------------------------------------
    def make_named_qubits(self) -> Dict[str, QuantumRegister]:
        qregs: Dict[str, QuantumRegister] = {}
        for q in self.data + self.az + self.ax:
            qregs[q] = QuantumRegister(1, q)
        return qregs

    def make_named_syndrome_bits(self) -> Dict[str, ClassicalRegister]:
        cregs: Dict[str, ClassicalRegister] = {}
        for i in range(4):
            cregs[f"sZ{i}"] = ClassicalRegister(1, f"sZ{i}")
        for i in range(4):
            cregs[f"sX{i}"] = ClassicalRegister(1, f"sX{i}")
        return cregs

    @staticmethod
    def Q(qregs: Dict[str, QuantumRegister], name: str):
        return qregs[name][0]

    @staticmethod
    def C(cregs: Dict[str, ClassicalRegister], name: str):
        return cregs[name][0]

    # ------------------------------------------------------------------
    # Stabilizer measurement primitives
    # ------------------------------------------------------------------
    def measure_Z_check(self, qc: QuantumCircuit, qregs: Dict[str, QuantumRegister],
                        anc: str, data: Tuple[str, ...], cbit, label: Optional[str]=None):
        if label:
            qc.barrier(label=label)
        qc.reset(self.Q(qregs, anc))
        for d in data:
            qc.cx(self.Q(qregs, d), self.Q(qregs, anc))
        qc.measure(self.Q(qregs, anc), cbit)

    def measure_X_check(self, qc: QuantumCircuit, qregs: Dict[str, QuantumRegister],
                        anc: str, data: Tuple[str, ...], cbit, label: Optional[str]=None):
        if label:
            qc.barrier(label=label)
        qc.reset(self.Q(qregs, anc))
        qc.h(self.Q(qregs, anc))
        for d in data:
            qc.cx(self.Q(qregs, anc), self.Q(qregs, d))
        qc.h(self.Q(qregs, anc))
        qc.measure(self.Q(qregs, anc), cbit)

    # ------------------------------------------------------------------
    # Full round builder
    # ------------------------------------------------------------------
    def build_one_round(self, measure_both: bool=True) -> QuantumCircuit:
        qregs = self.make_named_qubits()
        cregs = self.make_named_syndrome_bits()
        qc = QuantumCircuit(*qregs.values(), *cregs.values(), name="Surface17_one_round")

        for i, stab in enumerate(self.z_checks):
            self.measure_Z_check(qc, qregs, stab.anc, stab.data, self.C(cregs, f"sZ{i}"), label=stab.name)

        if measure_both:
            for i, stab in enumerate(self.x_checks):
                self.measure_X_check(qc, qregs, stab.anc, stab.data, self.C(cregs, f"sX{i}"), label=stab.name)

        qc.barrier(label="END_ROUND")
        return qc

    # ------------------------------------------------------------------
    # Error injection helper
    # ------------------------------------------------------------------
    def inject_error_then_round(self, err_type: str="X", err_data: str="d1", measure_both: bool=True) -> QuantumCircuit:
        qregs = self.make_named_qubits()
        cregs = self.make_named_syndrome_bits()
        qc = QuantumCircuit(*qregs.values(), *cregs.values(), name="Inject_then_round")

        qc.barrier(label="INJECT")
        if err_type.upper() == "X":
            qc.x(self.Q(qregs, err_data))
        elif err_type.upper() == "Z":
            qc.z(self.Q(qregs, err_data))
        else:
            raise ValueError("err_type must be 'X' or 'Z'")

        qc.compose(self.build_one_round(measure_both=measure_both), inplace=True)
        return qc

    # ------------------------------------------------------------------
    # Syndrome parsing helpers (counts keys may contain spaces)
    # ------------------------------------------------------------------
    @staticmethod
    def clean01(s: str) -> str:
        return re.sub(r"[^01]", "", s)

    @classmethod
    def parse_syndrome_from_key(cls, key: str) -> Tuple[str, str, List[int], List[int], List[int]]:
        raw = key
        c = cls.clean01(key)
        tail = c[-8:]
        bits = [int(b) for b in tail]          # [sZ0..sZ3,sX0..sX3]
        sZ = bits[0:4]
        sX = bits[4:8]
        return raw, tail, sZ, sX, bits

    # ------------------------------------------------------------------
    # Toy single-error decoder (for demos)
    # ------------------------------------------------------------------
    def _build_single_X_lookup(self) -> Dict[Tuple[int,int,int,int], str]:
        lookup: Dict[Tuple[int,int,int,int], str] = {}
        for d in self.data:
            pattern = tuple(1 if d in stab.data else 0 for stab in self.z_checks)
            lookup[pattern] = d
        return lookup

    def _build_single_Z_lookup(self) -> Dict[Tuple[int,int,int,int], str]:
        lookup: Dict[Tuple[int,int,int,int], str] = {}
        for d in self.data:
            pattern = tuple(1 if d in stab.data else 0 for stab in self.x_checks)
            lookup[pattern] = d
        return lookup

    def toy_decode_single_X(self, sZ: List[int]) -> Optional[str]:
        return self._x_lookup.get(tuple(sZ), None)

    def toy_decode_single_Z(self, sX: List[int]) -> Optional[str]:
        return self._z_lookup.get(tuple(sX), None)
