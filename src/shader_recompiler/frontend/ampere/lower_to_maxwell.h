// SPDX-FileCopyrightText: Copyright 2022 Collabora Ltd.
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.
// The accompanying MIT notice is in LICENSES/MIT.txt.
#pragma once

#include <array>
#include <span>
#include <string_view>
#include <vector>
#include "common/common_types.h"

namespace Shader::Ampere {

struct Instruction {
    std::array<u32, 4> words;

    u32 Bits(u32 offset, u32 count) const {
        u64 bits = words[offset / 32];
        if (offset / 32 + 1 < words.size()) {
            bits |= u64{words[offset / 32 + 1]} << 32;
        }
        return static_cast<u32>((bits >> (offset % 32)) & ((u64{1} << count) - 1));
    }
};
static_assert(sizeof(Instruction) == 16);

struct LoweredProgram {
    std::vector<u64> instructions;
    std::vector<u32> source_instruction_offsets;
    u32 source_size{};
    u32 required_texture_buffer = ~0U;
    u32 error_offset{};
    std::string_view error;

    explicit operator bool() const {
        return error.empty();
    }
};

// Lower only operations whose semantics are representable by the existing
// Maxwell frontend. These are new instructions, not reinterpretations of the
// native bytes. Scheduling/control words are added by the environment adapter.
LoweredProgram LowerToMaxwell(std::span<const Instruction> code);

} // namespace Shader::Ampere
