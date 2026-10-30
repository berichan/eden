// SPDX-FileCopyrightText: Copyright 2022 Collabora Ltd.
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <bit>
#include <string_view>
#include "video_core/turing_macro.h"

namespace Tegra {

TuringMacroEngine::Instruction TuringMacroEngine::Decode(const EncodedInstruction& words) {
    // Upload order is high word first. Several fields straddle the 32-bit word boundaries.
    const auto field = [&](u32 first, u32 count) {
        const auto index = first / 32;
        u64 value = u64{words[2 - index]} >> (first % 32);
        if (first % 32 + count > 32) {
            value |= u64{words[1 - index]} << (32 - first % 32);
        }
        return static_cast<u32>(value & ((u64{1} << count) - 1));
    };
    Instruction inst;
    inst.end_next = field(0, 1) != 0;
    inst.predicate_mode = static_cast<u8>(field(1, 4));
    inst.predicate = static_cast<u8>(field(5, 5));
    for (u32 slot = 0; slot < 2; ++slot) {
        const u32 base = slot == 0 ? 10 : 46;
        inst.alu[slot] =
            Alu{static_cast<Operation>(field(base, 5)), static_cast<u8>(field(base + 5, 5)),
                static_cast<u8>(field(base + 10, 5)), static_cast<u8>(field(base + 15, 5))};
        inst.immediate[slot] = static_cast<u16>(field(base + 20, 16));
        inst.output[slot] = Output{static_cast<u8>(field(82 + slot * 7, 3)),
                                   static_cast<u8>(field(85 + slot * 7, 4))};
    }
    return inst;
}

bool TuringMacroEngine::SetInstructionPointer(u32 instruction) {
    if (instruction >= InstructionCount) {
        return false;
    }
    upload_word = instruction * 3;
    return true;
}

bool TuringMacroEngine::UploadInstruction(u32 word) {
    if (upload_word >= InstructionCount * 3) {
        return false;
    }
    const u32 instruction = upload_word / 3;
    const u32 slot = upload_word % 3;
    code[instruction][slot] = word;
    written[instruction] |= u8{1} << slot;
    ++upload_word;
    return true;
}

bool TuringMacroEngine::ReadData(u32 index, u32& value) const {
    if (index >= data.size()) {
        return false;
    }
    value = data[index];
    return true;
}

bool TuringMacroEngine::WriteData(u32 index, u32 value) {
    if (index >= data.size()) {
        return false;
    }
    data[index] = value;
    return true;
}

class TuringMacroEngine::Invocation {
public:
    Invocation(TuringMacroEngine& engine_, const Host& host_, u32 start, u64 limit_)
        : engine{engine_}, host{host_}, limit{limit_}, pc{start} {}

    Result Run() {
        bool end_next{};
        while (result.error == Error::None) {
            result.program_counter = pc;
            if (host.should_stop && host.should_stop(host.context)) {
                Fail(Error::Cancelled);
                break;
            }
            if (result.instructions == limit) {
                Fail(Error::InstructionLimit);
                break;
            }
            if (pc >= InstructionCount) {
                Fail(Error::InvalidProgramCounter);
                break;
            }
            if (engine.written[pc] != 7) {
                Fail(Error::IncompleteInstruction);
                break;
            }
            inst = Decode(engine.code[pc]);
            ++result.instructions;
            next_pc = pc + 1;
            branch_cancels_end = false;
            if (inst.alu[0].operation == Operation::State &&
                inst.alu[1].operation == Operation::State) {
                Fail(Error::InvalidInstruction);
                break;
            }
            LoadParameters();
            const auto mask = PredicateMask();
            for (u32 slot = 0; slot < 2 && result.error == Error::None; ++slot) {
                if ((mask & (1U << slot)) != 0) {
                    EvaluateAlu(slot);
                }
            }
            for (u32 slot = 0; slot < 2 && result.error == Error::None; ++slot) {
                if ((mask & (4U << slot)) != 0) {
                    EvaluateOutput(slot);
                }
            }
            if (result.error != Error::None || end_next || stop) {
                break;
            }
            end_next = inst.end_next && !branch_cancels_end;
            if (loop_count != 0 && pc == loop_end) {
                --loop_count;
                next_pc = loop_start + 1;
            }
            pc = next_pc;
        }
        return result;
    }

private:
    void Fail(Error error) {
        if (result.error == Error::None) {
            result.error = error;
        }
    }

    void LoadParameters() {
        std::array<bool, 2> needed{};
        const auto source = [&](u8 reg) {
            if (reg == Load0 || reg == Load1) {
                needed[reg - Load0] = true;
            }
        };
        source(inst.predicate);
        for (u32 slot = 0; slot < 2; ++slot) {
            source(inst.alu[slot].source_a);
            source(inst.alu[slot].source_b);
            if (inst.output[slot].method == 3 || inst.output[slot].emit == 3) {
                needed[0] = true;
            }
            if (inst.output[slot].method == 4 || inst.output[slot].emit == 4) {
                needed[1] = true;
            }
        }
        if (needed[1] && !needed[0]) {
            Fail(Error::InvalidInstruction);
            return;
        }
        // Parameter consumption precedes predication, including disabled ALU/output slots.
        for (u32 slot = 0; slot < 2; ++slot) {
            if (needed[slot]) {
                if (!host.load_parameter || !host.load_parameter(host.context, loads[slot])) {
                    Fail(Error::ParameterUnderflow);
                    return;
                }
                ++result.parameters;
            }
        }
    }

    u32 Register(u8 reg, u32 slot) {
        if (reg < Zero) {
            if ((defined & (1U << reg)) == 0) {
                Fail(Error::UndefinedRegister);
            }
            return registers[reg];
        }
        switch (reg) {
        case Zero:
            return 0;
        case Immediate:
        case PairedImmediate:
            if (slot >= 2) {
                Fail(Error::InvalidRegister);
                return 0;
            }
            return static_cast<u32>(static_cast<s32>(
                static_cast<s16>(inst.immediate[reg == Immediate ? slot : 1 - slot])));
        case Immediate32:
            return (u32{inst.immediate[0]} << 16) | inst.immediate[1];
        case Load0:
            return loads[0];
        case Load1:
            return loads[1];
        default:
            Fail(Error::InvalidRegister);
            return 0;
        }
    }

    u8 PredicateMask() {
        constexpr std::array<std::string_view, 16> modes{
            "UUUU", "TTTT", "FFFF", "TTUU", "FFUU", "TFUU", "TUUU", "FUUU",
            "UUTT", "UUTF", "UUTU", "UUFT", "UUFF", "UUFU", "UUUT", "UUUF"};
        if (inst.predicate_mode == 0) {
            return 15;
        }
        const bool predicate = Register(inst.predicate, 2) != 0;
        u8 mask{};
        for (u32 slot = 0; slot < 4; ++slot) {
            if (modes[inst.predicate_mode][slot] != (predicate ? 'T' : 'F')) {
                mask |= u8{1} << slot;
            }
        }
        return mask;
    }

    void Store(u8 reg, u32 value) {
        if (reg < Zero) {
            registers[reg] = value;
            defined |= 1U << reg;
        } else if (reg != Zero) {
            Fail(Error::InvalidRegister);
        }
    }

    static bool Compare(Operation op, u32 x, u32 y) {
        switch (op) {
        case Operation::Less:
        case Operation::BranchLess:
            return static_cast<s32>(x) < static_cast<s32>(y);
        case Operation::LessUnsigned:
        case Operation::BranchLessUnsigned:
            return x < y;
        case Operation::LessEqual:
        case Operation::BranchLessEqual:
            return static_cast<s32>(x) <= static_cast<s32>(y);
        case Operation::LessEqualUnsigned:
        case Operation::BranchLessEqualUnsigned:
            return x <= y;
        default:
            return x == y;
        }
    }

    void EvaluateAlu(u32 slot) {
        const auto& alu = inst.alu[slot];
        const u32 x = Register(alu.source_a, slot);
        const u32 y = Register(alu.source_b, slot);
        if (result.error != Error::None) {
            return;
        }
        u32 value{};
        switch (alu.operation) {
        case Operation::Add:
            value = x + y;
            carry = value < x;
            break;
        case Operation::AddCarry:
        case Operation::SubtractBorrow:
            if (slot != 1 ||
                inst.alu[0].operation !=
                    (alu.operation == Operation::AddCarry ? Operation::Add : Operation::Subtract)) {
                Fail(Error::InvalidInstruction);
                return;
            }
            value = alu.operation == Operation::AddCarry ? x + y + carry : x - y - carry;
            break;
        case Operation::Subtract:
            value = x - y;
            carry = value > x;
            break;
        case Operation::Multiply:
        case Operation::MultiplyUnsigned: {
            const u64 a = alu.operation == Operation::Multiply
                              ? static_cast<u64>(static_cast<s64>(static_cast<s32>(x)))
                              : u64{x};
            const u64 b = alu.operation == Operation::Multiply
                              ? static_cast<u64>(static_cast<s64>(static_cast<s32>(y)))
                              : u64{y};
            const u64 product = a * b;
            value = static_cast<u32>(product);
            carry = static_cast<u32>(product >> 32);
            break;
        }
        case Operation::MultiplyHigh:
            if (alu.source_a != Zero || alu.source_b != Zero) {
                Fail(Error::InvalidInstruction);
                return;
            }
            value = carry;
            break;
        case Operation::Extended:
            if (x != 0x1000 || y != 1) {
                Fail(Error::UnsupportedInstruction);
                return;
            }
            if (!host.barrier || !host.barrier(host.context)) {
                Fail(Error::BarrierFailed);
                return;
            }
            break;
        case Operation::CountLeadingZeros:
            value = static_cast<u32>(std::countl_zero(x));
            break;
        case Operation::ShiftLeft:
            value = x << (y & 31);
            break;
        case Operation::ShiftRight:
            value = x >> (y & 31);
            break;
        case Operation::ShiftRightSigned:
            value = static_cast<u32>(static_cast<s32>(x) >> (y & 31));
            break;
        case Operation::And:
            value = x & y;
            break;
        case Operation::Nand:
            value = ~(x & y);
            break;
        case Operation::Or:
            value = x | y;
            break;
        case Operation::Xor:
            value = x ^ y;
            break;
        case Operation::Merge: {
            const u32 immediate = inst.immediate[slot];
            const u32 destination = immediate >> 10;
            const u32 bits = (immediate >> 5) & 31;
            const u32 source = immediate & 31;
            if (destination >= 32 && bits != 0) {
                Fail(Error::UnsupportedInstruction);
                return;
            }
            const u64 mask = (u64{1} << bits) - 1;
            value = static_cast<u32>((u64{x} & ~(mask << destination)) |
                                     (((u64{y} >> source) & mask) << destination));
            break;
        }
        case Operation::Less:
        case Operation::LessUnsigned:
        case Operation::LessEqual:
        case Operation::LessEqualUnsigned:
        case Operation::Equal:
            value = Compare(alu.operation, x, y) ? ~u32{0} : 0;
            break;
        case Operation::State:
            if (!host.read_state ||
                !host.read_state(host.context, static_cast<u16>((x + y) * 4), value)) {
                Fail(Error::StateReadFailed);
                return;
            }
            break;
        case Operation::Loop:
            if (loop_count != 0 || alu.destination != Zero || alu.source_b != Zero ||
                inst.immediate[slot] < 2 || pc + inst.immediate[slot] > InstructionCount) {
                Fail(Error::InvalidInstruction);
                return;
            }
            loop_start = pc;
            loop_end = pc + inst.immediate[slot] - 1;
            // Uploaded programs use a zero count to skip parameter-consuming bodies entirely.
            if (x == 0) {
                next_pc = loop_end + 1;
            } else {
                loop_count = x - 1;
            }
            break;
        case Operation::Jump:
            if (alu.destination != Zero || alu.source_a != Zero || alu.source_b != Zero ||
                (inst.immediate[slot] & 0x8000) == 0) {
                Fail(Error::InvalidInstruction);
                return;
            }
            Branch(pc + (inst.immediate[slot] & 0x7fff));
            break;
        case Operation::BranchLess:
        case Operation::BranchLessUnsigned:
        case Operation::BranchLessEqual:
        case Operation::BranchLessEqualUnsigned:
        case Operation::BranchEqual: {
            if (alu.destination != Zero) {
                Fail(Error::InvalidInstruction);
                return;
            }
            const bool expected = (inst.immediate[slot] & 0x8000) != 0;
            if (Compare(alu.operation, x, y) == expected) {
                // Native loop back edges use bit14 to suppress this instruction's
                // fallthrough termination marker, not a previously armed delay-slot exit.
                branch_cancels_end = (inst.immediate[slot] & 0x4000) != 0;
                s32 offset = inst.immediate[slot] & 0x1fff;
                if ((offset & 0x1000) != 0) {
                    offset -= 0x2000;
                }
                if (offset == -0x1000) {
                    stop = true;
                } else {
                    Branch(static_cast<s64>(pc) + offset);
                }
            }
            break;
        }
        case Operation::DataRead:
            if (alu.source_b != Zero) {
                Fail(Error::InvalidInstruction);
                return;
            }
            if (!engine.ReadData(x, value)) {
                Fail(Error::InvalidDataAddress);
                return;
            }
            break;
        case Operation::DataWrite:
            if (alu.destination != Zero) {
                Fail(Error::InvalidInstruction);
                return;
            }
            if (!engine.WriteData(x, y)) {
                Fail(Error::InvalidDataAddress);
                return;
            }
            break;
        }
        alu_results[slot] = value;
        Store(alu.destination, value);
    }

    void Branch(s64 destination) {
        if (destination < 0 || destination >= static_cast<s64>(InstructionCount)) {
            Fail(Error::InvalidBranch);
        } else {
            next_pc = static_cast<u32>(destination);
        }
    }

    u32 OutputValue(u8 selector) {
        switch (selector) {
        case 1:
            return alu_results[0];
        case 2:
            return alu_results[1];
        case 3:
            return loads[0];
        case 4:
            return loads[1];
        case 5:
            return inst.immediate[0];
        case 6:
            return inst.immediate[1];
        case 8:
            return inst.immediate[0] >> 12;
        case 9:
            return inst.immediate[1] >> 12;
        case 10:
            return (u32{inst.immediate[0]} << 16) | inst.immediate[1];
        default:
            Fail(Error::InvalidInstruction);
            return 0;
        }
    }

    void EvaluateOutput(u32 slot) {
        const auto& out = inst.output[slot];
        if (out.method != 0) {
            const u32 value = OutputValue(out.method);
            method = static_cast<u16>((value & 0xfff) << 2);
            increment = (value >> 12) & 15;
        }
        if (out.emit != 0 && result.error == Error::None) {
            const u32 value = OutputValue(out.emit);
            if (result.error != Error::None) {
                return;
            }
            if (!host.emit_method || !host.emit_method(host.context, method, value)) {
                Fail(Error::MethodEmitFailed);
                return;
            }
            ++result.methods;
            method = static_cast<u16>(method + increment * 4);
        }
    }

    TuringMacroEngine& engine;
    const Host& host;
    u64 limit;
    Result result;
    Instruction inst;
    std::array<u32, 24> registers{};
    std::array<u32, 2> loads{};
    std::array<u32, 2> alu_results{};
    u32 defined{};
    u32 carry{};
    u32 pc;
    u32 next_pc{};
    u16 method{};
    u32 increment{};
    u32 loop_count{};
    u32 loop_start{};
    u32 loop_end{};
    bool stop{};
    bool branch_cancels_end{};
};

TuringMacroEngine::Result TuringMacroEngine::Execute(u32 start, const Host& host,
                                                     u64 instruction_limit) {
    return Invocation{*this, host, start, instruction_limit}.Run();
}

} // namespace Tegra
