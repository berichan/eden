// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>
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

// A pair may use a 16-byte backend packet. An eight-byte-aligned offset and
// a 64-byte-aligned bound cannot straddle the packet at that bound. Prove
// alignment without rounding the guest's address or assuming loop invariants.
class OffsetAlignment {
public:
    u32 Bits(const IR::Value& value) {
        const u32 bits = Visit(value, 0);
        return invalid ? 0 : bits;
    }

private:
    u32 Visit(const IR::Value& value, u32 depth) {
        if (depth >= 64 || nodes >= 512) {
            invalid = true;
            return 0;
        }
        ++nodes;
        if (value.IsEmpty()) {
            return 0;
        }
        // IsImmediate() follows identities itself, so handle them only after
        // recording the bounded visit and checking for a cycle.
        if (!value.IsIdentity() && value.IsImmediate()) {
            return value.Type() == IR::Type::U32 ? static_cast<u32>(std::countr_zero(value.U32()))
                                                 : 0;
        }
        const auto* instruction = value.Inst();
        if (const auto found = known.find(instruction); found != known.end()) {
            return found->second;
        }
        if (!visiting.insert(instruction).second) {
            invalid = true;
            return 0;
        }
        const auto arg = [&](size_t index) { return Visit(instruction->Arg(index), depth + 1); };
        const auto constant_bits = [&](size_t index) -> u32 {
            const auto constant = ResolveIdentity(instruction->Arg(index));
            return constant && !constant->IsEmpty() && constant->IsImmediate() &&
                           constant->Type() == IR::Type::U32
                       ? static_cast<u32>(std::countr_zero(constant->U32()))
                       : 0;
        };
        u32 bits{};
        switch (instruction->GetOpcode()) {
        case IR::Opcode::Identity:
            bits = arg(0);
            break;
        case IR::Opcode::BitwiseAnd32:
            // This operand alone proves sufficient alignment, regardless of
            // any loop-dependent value on the other side of the operation.
            bits = (std::max)(constant_bits(0), constant_bits(1));
            if (bits >= 3) {
                break;
            }
            bits = (std::max)(arg(0), arg(1));
            break;
        case IR::Opcode::IAdd32:
        case IR::Opcode::ISub32:
        case IR::Opcode::BitwiseOr32:
        case IR::Opcode::BitwiseXor32:
            bits = (std::min)(arg(0), arg(1));
            break;
        case IR::Opcode::IMul32:
            bits = (std::max)(constant_bits(0), constant_bits(1));
            if (bits >= 3) {
                break;
            }
            bits = (std::min)(32U, arg(0) + arg(1));
            break;
        case IR::Opcode::ShiftLeftLogical32: {
            const auto shift = ResolveIdentity(instruction->Arg(1));
            if (shift && !shift->IsEmpty() && shift->IsImmediate() &&
                shift->Type() == IR::Type::U32 && shift->U32() < 32) {
                bits = shift->U32();
                if (bits < 3) {
                    bits = (std::min)(32U, arg(0) + bits);
                }
            }
            break;
        }
        case IR::Opcode::SelectU32:
            bits = (std::min)(arg(1), arg(2));
            break;
        case IR::Opcode::Phi:
            if (instruction->NumArgs()) {
                bits = 32;
                for (size_t index = 0; index < instruction->NumArgs(); ++index) {
                    bits = (std::min)(bits, arg(index));
                }
            }
            break;
        default:
            break;
        }
        visiting.erase(instruction);
        known.emplace(instruction, bits);
        return bits;
    }

    std::unordered_map<const IR::Inst*, u32> known;
    std::unordered_set<const IR::Inst*> visiting;
    u32 nodes{};
    bool invalid{};
};

std::string DescribeOffset(const IR::Value& value, u32 depth) {
    const auto resolved = ResolveIdentity(value);
    if (!resolved)
        return "identity-limit";
    if (resolved->IsEmpty())
        return "empty";
    if (resolved->IsImmediate())
        return resolved->Type() == IR::Type::U32 ? fmt::format("{:#x}", resolved->U32())
                                                 : "literal";
    const auto* instruction = resolved->Inst();
    std::string result{IR::NameOf(instruction->GetOpcode())};
    if (depth) {
        for (u32 index = 0; index < (std::min)(instruction->NumArgs(), size_t{2}); ++index)
            result += fmt::format(" [{}]", DescribeOffset(instruction->Arg(index), depth - 1));
    }
    return result;
}
} // namespace

void GuardConstantBufferLoads(IR::Program& program, std::span<const u32> bound_sizes) {
    std::vector<GuardedLoad> guarded;
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
                    continue;
                }
                if (opcode != IR::Opcode::GetCbufU32x2 || size % 64 ||
                    OffsetAlignment{}.Bits(offset) < 3) {
                    throw RuntimeError("Unsupported native narrow/vector constant-buffer bound: "
                                       "opcode={}, bank={}, bound={:#x}, offset={}",
                                       IR::NameOf(opcode), bank.U32(), size,
                                       DescribeOffset(offset, 2));
                }
                guarded.push_back({block, &instruction, size});
                continue;
            }
            if (offset.IsImmediate() && (offset.U32() & ~3U) <= size - 4) {
                continue;
            }
            if (size % 64) {
                throw RuntimeError("Native dynamic constant-buffer bound is not 64-byte aligned");
            }
            guarded.push_back({block, &instruction, size});
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
            const auto valid = ir.ILessThanEqual(offset, ir.Imm32(load.size - 8), false);
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
}

} // namespace Shader::Ampere
