// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/constant_buffer_bounds.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Ampere {
namespace {
u32 LoadSize(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::GetCbufU8:
    case IR::Opcode::GetCbufS8:
        return 1;
    case IR::Opcode::GetCbufU16:
    case IR::Opcode::GetCbufS16:
        return 2;
    case IR::Opcode::GetCbufU32:
    case IR::Opcode::GetCbufF32:
        return 4;
    case IR::Opcode::GetCbufU32x2:
        return 8;
    default:
        return 0;
    }
}

struct GuardedLoad {
    IR::Block* block;
    IR::Inst* instruction;
    u32 size;
};

std::optional<IR::Value> ResolveIdentity(IR::Value value) {
    for (u32 depth = 0; depth < 64; ++depth) {
        if (!value.IsIdentity()) {
            return value.Resolve();
        }
        value = value.Inst()->Arg(0);
    }
    return std::nullopt;
}

class UnsignedOffsetBound {
public:
    u32 Upper(const IR::Value& value, u32 depth = 0) {
        if (depth >= 64 || value.IsEmpty())
            return Unknown;
        if (!value.IsIdentity() && value.IsImmediate())
            return value.Type() == IR::Type::U32 ? value.U32() : Unknown;
        const auto* instruction = value.Inst();
        if (const auto found = bounds.find(instruction); found != bounds.end())
            return found->second.value_or(Unknown);
        if (bounds.size() >= 512)
            return Unknown;
        // An in-progress node is unknown. In particular, do not assume a phi's
        // seed bounds later iterations of its recurrence.
        bounds.emplace(instruction, std::nullopt);
        const u32 upper = Evaluate(*instruction, depth + 1);
        bounds[instruction] = upper;
        return upper;
    }

private:
    static constexpr u32 Unknown = std::numeric_limits<u32>::max();

    static std::optional<u32> Immediate(const IR::Value& value) {
        const auto resolved = ResolveIdentity(value);
        if (resolved && !resolved->IsEmpty() && resolved->IsImmediate() &&
            resolved->Type() == IR::Type::U32)
            return resolved->U32();
        return std::nullopt;
    }

    u32 Evaluate(const IR::Inst& instruction, u32 depth) {
        const auto upper = [&](size_t index) { return Upper(instruction.Arg(index), depth); };
        switch (instruction.GetOpcode()) {
        case IR::Opcode::Identity:
            return upper(0);
        case IR::Opcode::UMin32:
        case IR::Opcode::BitwiseAnd32:
            // Either operation cannot exceed either operand. An immediate
            // clamp/mask bounds even a cyclic or otherwise opaque input.
            if (const auto literal = Immediate(instruction.Arg(0)))
                return *literal;
            if (const auto literal = Immediate(instruction.Arg(1)))
                return *literal;
            return (std::min)(upper(0), upper(1));
        case IR::Opcode::ShiftLeftLogical32: {
            const auto shift = Immediate(instruction.Arg(1));
            if (!shift || *shift >= 32)
                return Unknown;
            const u32 source = upper(0);
            return source <= (Unknown >> *shift) ? source << *shift : Unknown;
        }
        case IR::Opcode::IAdd32: {
            const u64 sum = u64{upper(0)} + upper(1);
            return sum <= Unknown ? static_cast<u32>(sum) : Unknown;
        }
        case IR::Opcode::IMul32: {
            const u64 product = u64{upper(0)} * upper(1);
            return product <= Unknown ? static_cast<u32>(product) : Unknown;
        }
        case IR::Opcode::SelectU32:
            return (std::max)(upper(1), upper(2));
        default:
            return Unknown;
        }
    }

    std::unordered_map<const IR::Inst*, std::optional<u32>> bounds;
};

u32 UploadPrefix(u32 upper, u32 width, u32 bound) {
    // Guards can only read zero or an in-range address. Include the containing
    // vec4 packet even on hosts that implement the load with scalar accesses.
    const u32 last = (std::min)(upper, bound - width) & ~(width - 1);
    return (last + width + 15) & ~15U;
}

std::string DescribeOffset(const IR::Value& value) {
    // Record shared nodes and back edges explicitly. A recursive tree hides
    // induction inputs and can expand exponentially through nested phis.
    constexpr size_t MaxNodes = 64, MaxArgs = 16;
    std::vector<const IR::Inst*> nodes;
    std::unordered_map<const IR::Inst*, size_t> indices;
    const auto reference = [&](const IR::Value& argument) -> std::string {
        if (argument.IsEmpty())
            return "empty";
        if (!argument.IsIdentity() && argument.IsImmediate()) {
            if (argument.Type() == IR::Type::U32)
                return fmt::format("{:#x}", argument.U32());
            if (argument.Type() == IR::Type::F32)
                return fmt::format("f32bits({:#x})", std::bit_cast<u32>(argument.F32()));
            return "literal";
        }
        const auto* instruction = argument.Inst();
        if (const auto found = indices.find(instruction); found != indices.end())
            return fmt::format("%{}", found->second);
        if (nodes.size() == MaxNodes)
            return "node-limit";
        const size_t index = nodes.size();
        indices.emplace(instruction, index);
        nodes.push_back(instruction);
        return fmt::format("%{}", index);
    };
    std::string result = reference(value);
    for (size_t index = 0; index < nodes.size(); ++index) {
        const auto* instruction = nodes[index];
        result += fmt::format("; %{}={}", index, IR::NameOf(instruction->GetOpcode()));
        for (size_t arg = 0; arg < (std::min)(instruction->NumArgs(), MaxArgs); ++arg)
            result += fmt::format(" [{}]", reference(instruction->Arg(arg)));
        if (instruction->NumArgs() > MaxArgs)
            result += " [argument-limit]";
    }
    return result;
}
} // namespace

void GuardConstantBufferLoads(IR::Program& program, std::span<const u32> bound_sizes) {
    std::vector<GuardedLoad> guarded;
    UnsignedOffsetBound offset_bound;
    std::array<u32, Info::MAX_CBUFS> used_sizes{};
    std::array<bool, Info::MAX_CBUFS> touched{};
    u32 reported_driver_offsets{};
    for (IR::Block* block : program.blocks) {
        for (auto& instruction : block->Instructions()) {
            const auto opcode = instruction.GetOpcode();
            const u32 width = LoadSize(opcode);
            if (!width) {
                continue;
            }
            const auto bank = instruction.Arg(0);
            const auto offset = instruction.Arg(1);
            if (!bank.IsImmediate() || bank.U32() >= Info::MAX_CBUFS ||
                bank.U32() >= bound_sizes.size()) {
                throw RuntimeError("Unsupported native constant-buffer bank");
            }
            const u32 size = bound_sizes[bank.U32()];
            if (size < 4 || size > 0x10000) {
                throw RuntimeError("Unsupported native constant-buffer bound");
            }
            const bool scalar =
                opcode == IR::Opcode::GetCbufU32 || opcode == IR::Opcode::GetCbufF32;
            if (!scalar) {
                // The backend may implement narrow/pair loads with a 16-byte
                // vector. Require that complete packet in bounds as well.
                const auto resolved = ResolveIdentity(offset);
                const bool immediate = resolved && !resolved->IsEmpty() &&
                                       resolved->IsImmediate() && resolved->Type() == IR::Type::U32;
                if (immediate && resolved->U32() % width == 0 && size >= width &&
                    resolved->U32() <= size - width && size >= 16 &&
                    (resolved->U32() & ~15U) <= size - 16) {
                    // This static load needs no address guard.
                } else if (opcode != IR::Opcode::GetCbufU32x2 || size % 64) {
                    throw RuntimeError("Unsupported native narrow/vector constant-buffer bound: "
                                       "opcode={}, bank={}, bound={:#x}, offset={}",
                                       IR::NameOf(opcode), bank.U32(), size,
                                       DescribeOffset(offset));
                } else {
                    guarded.push_back({block, &instruction, size});
                }
            } else if (!(offset.IsImmediate() && (offset.U32() & ~3U) <= size - 4)) {
                if (size % 64)
                    throw RuntimeError(
                        "Native dynamic constant-buffer bound is not 64-byte aligned");
                guarded.push_back({block, &instruction, size});
            }
            // Only the aligned native-binding contract admits upload narrowing.
            // Preserve the existing small static-load contract independently.
            if (size % 64 == 0) {
                const u32 prefix = UploadPrefix(offset_bound.Upper(offset), width, size);
                touched[bank.U32()] = true;
                used_sizes[bank.U32()] = (std::max)(used_sizes[bank.U32()], prefix);
                if (bank.U32() == 1 && size == 0x10000 && prefix == size && !offset.IsImmediate() &&
                    reported_driver_offsets < 8) {
                    ++reported_driver_offsets;
                    LOG_DEBUG(HW_GPU,
                              "Native dynamic driver constant offset: stage={}, opcode={}, "
                              "bound={:#x}, offset={}",
                              static_cast<u32>(program.stage), IR::NameOf(opcode), size,
                              DescribeOffset(offset));
                }
            }
        }
    }
    // Validate the entire program before mutating it. A later unsupported load
    // must not leave an earlier partially transformed program behind.
    for (const auto& load : guarded) {
        auto& instruction = *load.instruction;
        IR::IREmitter ir{*load.block, IR::Block::InstructionList::s_iterator_to(instruction)};
        const IR::U32 bank{instruction.Arg(0)};
        const IR::U32 offset{instruction.Arg(1)};
        if (instruction.GetOpcode() == IR::Opcode::GetCbufU32x2) {
            // Register reuse across loops can obscure an otherwise naturally
            // aligned LDC.64 address in SSA. Check the actual address instead.
            // Eight-byte alignment also keeps either backend's 16-byte packet
            // inside the aligned binding. Misaligned native pair accesses are
            // outside this contract and yield zero; never round their address.
            const auto aligned = ir.IEqual(ir.BitwiseAnd(offset, ir.Imm32(7U)), ir.Imm32(0U));
            const auto in_bounds = ir.ILessThanEqual(offset, ir.Imm32(load.size - 8), false);
            const auto valid = ir.LogicalAnd(aligned, in_bounds);
            const IR::U32 safe_offset{ir.Select(valid, offset, ir.Imm32(0U))};
            const auto pair = ir.GetCbuf(bank, safe_offset, 64, false);
            const auto low = ir.Select(valid, ir.CompositeExtract(pair, 0), ir.Imm32(0U));
            const auto high = ir.Select(valid, ir.CompositeExtract(pair, 1), ir.Imm32(0U));
            instruction.ReplaceUsesWith(ir.CompositeConstruct(low, high));
            continue;
        }
        const auto aligned = ir.BitwiseAnd(offset, ir.Imm32(~3U));
        const auto valid = ir.ILessThanEqual(aligned, ir.Imm32(load.size - 4), false);
        const IR::U32 safe_offset{ir.Select(valid, aligned, ir.Imm32(0U))};
        if (instruction.GetOpcode() == IR::Opcode::GetCbufF32) {
            const auto value = ir.GetFloatCbuf(bank, safe_offset);
            instruction.ReplaceUsesWith(ir.Select(valid, value, ir.Imm32(0.f)));
        } else {
            const auto value = ir.GetCbuf(bank, safe_offset);
            instruction.ReplaceUsesWith(ir.Select(valid, value, ir.Imm32(0U)));
        }
    }
    for (u32 bank = 0; bank < used_sizes.size(); ++bank) {
        if (touched[bank])
            program.info.constant_buffer_used_sizes[bank] = used_sizes[bank];
    }
}

} // namespace Shader::Ampere
