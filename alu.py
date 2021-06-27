from enum import IntEnum
from typing import List

from nmigen import Elaboratable, Module, Signal
from nmigen.build import Platform
from nmigen.cli import main_parser, main_runner
from nmigen.sim import Simulator, Delay


class AluFunc(IntEnum):
    NONE = 0
    # Arithmetic
    ADD = 1
    SUB = 2

    # Logical
    NEG = 8
    AND = 9
    OR = 10
    ROL = 11  # Roll left
    ROR = 12  # Roll right
    LSL = 13
    LSR = 14
    ASR = 15
    CLC = 16  # clear carry
    CLV = 17  # clear overflow


class Alu(Elaboratable):
    def __init__(self, size=8):
        self.size = size
        self.A = Signal(size)
        self.B = Signal(size)
        self.OUT = Signal(size, reset=0)
        self.func = Signal(AluFunc)
        self.zero = Signal()
        self.carry = Signal(reset=0)
        self.negative = Signal()
        self.overflow = Signal(reset=0)
        self.signed = Signal()

    def elaborate(self, platform: Platform):
        m = Module()

        m.d.comb += self.signed.eq(self.negative ^ self.overflow)
        m.d.comb += self.zero.eq(self.OUT == 0)
        m.d.comb += self.negative.eq(self.OUT[self.size - 1])

        with m.Switch(self.func):
            with m.Case(AluFunc.ADD):
                m.d.comb += self.OUT.eq((self.A + self.B)[:self.size])
                m.d.comb += self.carry.eq((self.A + self.B)[self.size])
                m.d.comb += self.overflow.eq((self.A[self.size - 1] == self.B[self.size - 1])
                                             & (self.A[self.size - 1] != self.OUT[self.size - 1]))
            with m.Case(AluFunc.SUB):
                m.d.comb += self.OUT.eq((self.A - self.B)[:self.size])
                m.d.comb += self.carry.eq((self.A - self.B)[self.size])
                m.d.comb += self.overflow.eq((self.A[self.size - 1] != self.B[self.size - 1])
                                             & (self.A[self.size - 1] != self.OUT[self.size - 1]))
            with m.Case(AluFunc.NEG):
                m.d.comb += self.OUT.eq(~self.A)
            with m.Case(AluFunc.AND):
                m.d.comb += self.OUT.eq(self.A & self.B)
            with m.Case(AluFunc.OR):
                m.d.comb += self.OUT.eq(self.A | self.B)
            with m.Case(AluFunc.ROL):
                m.d.comb += self.OUT.eq((self.A << 1 | self.carry)[:self.size])
                m.d.comb += self.carry.eq(self.A[self.size - 1])
            with m.Case(AluFunc.ROR):
                m.d.comb += self.OUT.eq(self.A >> 1 | (self.carry << (self.size - 1))[:self.size])
                m.d.comb += self.carry.eq(self.A[0])
            with m.Case(AluFunc.LSL):
                m.d.comb += self.OUT.eq((self.A << 1)[:self.size])
                m.d.comb += self.carry.eq(self.A[self.size - 1])
            with m.Case(AluFunc.LSR):
                m.d.comb += self.OUT.eq((self.A >> 1)[:self.size])
                m.d.comb += self.carry.eq(self.A[0])
            with m.Case(AluFunc.ASR):
                m.d.comb += self.OUT.eq((self.A[:self.size - 1] >> 1
                                         | (self.A[self.size - 1] << (self.size - 1)))[:self.size])
                m.d.comb += self.carry.eq(self.A[0])
            with m.Case(AluFunc.CLC):
                m.d.comb += self.carry.eq(0)
            with m.Case(AluFunc.CLV):
                m.d.comb += self.overflow.eq(0)
            with m.Default():
                m.d.comb += self.OUT.eq(self.A)
        return m


if __name__ == "__main__":
    parser = main_parser()
    args = parser.parse_args()

    m = Module()
    m.submodules.alu = alu = Alu()

    sim = Simulator(m)

    a = alu.A
    b = alu.B
    func = alu.func
    out = alu.OUT


    def process():
        yield a.eq(0x01)
        yield b.eq(0x55)
        yield Delay(1e-6)
        yield func.eq(AluFunc.ADD)
        yield Delay(1e-6)
        yield func.eq(AluFunc.CLV)
        yield Delay(1e-6)
        yield func.eq(AluFunc.CLC)
        yield Delay(1e-6)
        yield func.eq(AluFunc.SUB)
        yield Delay(1e-6)
        yield func.eq(AluFunc.CLV)
        yield Delay(1e-6)
        yield func.eq(AluFunc.CLC)
        yield Delay(1e-6)
        yield func.eq(AluFunc.NEG)
        yield Delay(1e-6)
        yield func.eq(AluFunc.AND)
        yield Delay(1e-6)
        yield a.eq(0b01111111)
        yield b.eq(0b00000001)
        yield func.eq(AluFunc.ADD)
        yield Delay(1e-6)


    sim.add_process(process)
    with sim.write_vcd("test.vcd", "test.gtkw",
                       traces=[alu.A, alu.B, alu.func, alu.OUT, alu.zero, alu.carry, alu.overflow, alu.signed,
                               alu.negative]):
        sim.run()
