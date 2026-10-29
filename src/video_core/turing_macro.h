// SPDX-FileCopyrightText: Copyright 2022 Collabora Ltd.
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#pragma once

#include <array>
#include <cstddef>
#include "common/common_types.h"

namespace Tegra {

// Encoding and execution semantics adapted from Mesa's MIT-licensed TU104 MME simulator.
// https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/nouveau/mme
// The accompanying MIT notice is in LICENSES/MIT.txt.
// The owning graphics context serializes uploads, data access, and execution.
class TuringMacroEngine {
public:
    static constexpr size_t InstructionCount = 0x1000;
    static constexpr size_t DataCount = 0xc00;
    static constexpr u8 Zero = 24;
    static constexpr u8 Immediate = 25;
    static constexpr u8 PairedImmediate = 26;
    static constexpr u8 Immediate32 = 27;
    static constexpr u8 Load0 = 28;
    static constexpr u8 Load1 = 29;

    enum class Operation : u8 {
        Add,
        AddCarry,
        Subtract,
        SubtractBorrow,
        Multiply,
        MultiplyHigh,
        MultiplyUnsigned,
        Extended,
        CountLeadingZeros,
        ShiftLeft,
        ShiftRight,
        ShiftRightSigned,
        And,
        Nand,
        Or,
        Xor,
        Merge,
        Less,
        LessUnsigned,
        LessEqual,
        LessEqualUnsigned,
        Equal,
        State,
        Loop,
        Jump,
        BranchLess,
        BranchLessUnsigned,
        BranchLessEqual,
        BranchLessEqualUnsigned,
        BranchEqual,
        DataRead,
        DataWrite,
    };

    struct Alu {
        Operation operation{Operation::Add};
        u8 destination{Zero};
        u8 source_a{Zero};
        u8 source_b{Zero};
    };
    struct Output {
        u8 method{};
        u8 emit{};
    };
    struct Instruction {
        bool end_next{};
        u8 predicate_mode{};
        u8 predicate{Zero};
        std::array<u16, 2> immediate{};
        std::array<Alu, 2> alu{};
        std::array<Output, 2> output{};
    };
    using EncodedInstruction = std::array<u32, 3>;

    enum class Error {
        None,
        InvalidProgramCounter,
        IncompleteInstruction,
        InvalidInstruction,
        InvalidRegister,
        UndefinedRegister,
        ParameterUnderflow,
        InvalidBranch,
        InvalidDataAddress,
        UnsupportedInstruction,
        StateReadFailed,
        MethodEmitFailed,
        BarrierFailed,
        InstructionLimit,
        Cancelled,
    };
    struct Result {
        Error error{Error::None};
        u32 program_counter{};
        u64 instructions{};
        u64 parameters{};
        u64 methods{};

        bool Succeeded() const {
            return error == Error::None;
        }
    };

    struct Host {
        void* context{};
        bool (*load_parameter)(void*, u32&){};
        bool (*read_state)(void*, u16 byte_address, u32&){};
        bool (*emit_method)(void*, u16 byte_address, u32){};
        bool (*barrier)(void*){};
        bool (*should_stop)(void*){};
    };

    static Instruction Decode(const EncodedInstruction& words);
    bool SetInstructionPointer(u32 instruction);
    bool UploadInstruction(u32 word);
    bool ReadData(u32 index, u32& value) const;
    bool WriteData(u32 index, u32 value);
    Result Execute(u32 start, const Host& host, u64 instruction_limit);

private:
    class Invocation;
    std::array<EncodedInstruction, InstructionCount> code{};
    std::array<u8, InstructionCount> written{};
    std::array<u32, DataCount> data{};
    u32 upload_word{};
};

} // namespace Tegra
