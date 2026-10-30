// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <unordered_map>
#include <catch2/catch_test_macros.hpp>
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/constant_buffer_bounds.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace {
namespace IR = Shader::IR;
using Shader::Ampere::GuardConstantBufferLoads;

struct Fixture {
    Shader::ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    IR::IREmitter ir{block};

    Fixture() {
        program.blocks.push_back(&block);
    }
};

u32 Evaluate(IR::Block& block, u32 input, u32 bound) {
    std::unordered_map<const IR::Inst*, u32> values;
    const auto read = [&](const IR::Value& value) -> u32 {
        if (!value.IsImmediate()) {
            return values.at(value.Inst());
        }
        if (value.Type() == IR::Type::F32) {
            return std::bit_cast<u32>(value.F32());
        }
        return value.Type() == IR::Type::U1 ? static_cast<u32>(value.U1()) : value.U32();
    };
    u32 result{};
    for (const auto& instruction : block.Instructions()) {
        const auto arg = [&](size_t index) { return read(instruction.Arg(index)); };
        u32 value{};
        switch (instruction.GetOpcode()) {
        case IR::Opcode::GetRegister:
            value = input;
            break;
        case IR::Opcode::BitwiseAnd32:
            value = arg(0) & arg(1);
            break;
        case IR::Opcode::ULessThanEqual:
            value = arg(0) <= arg(1);
            break;
        case IR::Opcode::SelectU32:
        case IR::Opcode::SelectF32:
            value = arg(0) ? arg(1) : arg(2);
            break;
        case IR::Opcode::GetCbufU32:
        case IR::Opcode::GetCbufF32: {
            REQUIRE(arg(0) == 0);
            const u32 offset = arg(1) & ~3U;
            // Evaluate every load, including unused ones. A post-load zero
            // select alone must not conceal an unsafe memory access.
            REQUIRE(offset <= bound - 4);
            value = 0x3f000000U + offset;
            break;
        }
        case IR::Opcode::Identity:
            value = arg(0);
            break;
        case IR::Opcode::Reference:
            result = arg(0);
            continue;
        default:
            FAIL("Unexpected constant-buffer bounds instruction");
        }
        values.emplace(&instruction, value);
    }
    return result;
}

std::array<u32, 2> EvaluatePair(IR::Block& block, u32 input, u32 bound) {
    std::unordered_map<const IR::Inst*, u32> values;
    std::unordered_map<const IR::Inst*, std::array<u32, 2>> pairs;
    const auto read = [&](const IR::Value& value) -> u32 {
        if (!value.IsImmediate()) {
            return values.at(value.Inst());
        }
        if (value.Type() == IR::Type::F32) {
            return std::bit_cast<u32>(value.F32());
        }
        return value.Type() == IR::Type::U1 ? static_cast<u32>(value.U1()) : value.U32();
    };
    std::array<u32, 2> result{};
    for (const auto& instruction : block.Instructions()) {
        const auto arg = [&](size_t index) { return read(instruction.Arg(index)); };
        u32 value{};
        switch (instruction.GetOpcode()) {
        case IR::Opcode::Phi:
            // A loop-carried value is arbitrary here. The runtime guard must
            // protect the load without assuming an invariant for the recurrence.
            value = input;
            break;
        case IR::Opcode::GetRegister:
            value = input;
            break;
        case IR::Opcode::BitwiseAnd32:
            value = arg(0) & arg(1);
            break;
        case IR::Opcode::BitwiseOr32:
            value = arg(0) | arg(1);
            break;
        case IR::Opcode::BitwiseXor32:
            value = arg(0) ^ arg(1);
            break;
        case IR::Opcode::IAdd32:
            value = arg(0) + arg(1);
            break;
        case IR::Opcode::ISub32:
            value = arg(0) - arg(1);
            break;
        case IR::Opcode::IMul32:
            value = arg(0) * arg(1);
            break;
        case IR::Opcode::ShiftLeftLogical32:
            REQUIRE(arg(1) < 32);
            value = arg(0) << arg(1);
            break;
        case IR::Opcode::ULessThanEqual:
            value = arg(0) <= arg(1);
            break;
        case IR::Opcode::IEqual:
            value = arg(0) == arg(1);
            break;
        case IR::Opcode::LogicalAnd:
            value = arg(0) && arg(1);
            break;
        case IR::Opcode::SelectU32:
            value = arg(0) ? arg(1) : arg(2);
            break;
        case IR::Opcode::GetCbufU32x2: {
            REQUIRE(arg(0) == 0);
            const u32 offset = arg(1);
            REQUIRE(offset % 8 == 0);
            REQUIRE(offset <= bound - 8);
            // Also model the backend that fetches the containing vec4 packet.
            REQUIRE((offset & ~15U) <= bound - 16);
            pairs.emplace(&instruction, std::array{0x3f000000U + offset, 0x40000000U + offset + 4});
            continue;
        }
        case IR::Opcode::CompositeExtractU32x2:
            value = pairs.at(instruction.Arg(0).Inst()).at(arg(1));
            break;
        case IR::Opcode::CompositeConstructU32x2:
            pairs.emplace(&instruction, std::array{arg(0), arg(1)});
            continue;
        case IR::Opcode::Identity:
            if (instruction.Arg(0).Type() == IR::Type::U32x2) {
                pairs.emplace(&instruction, pairs.at(instruction.Arg(0).Inst()));
                continue;
            }
            value = arg(0);
            break;
        case IR::Opcode::BitCastU32F32:
        case IR::Opcode::BitCastF32U32:
            value = arg(0);
            break;
        case IR::Opcode::Reference:
            result = pairs.at(instruction.Arg(0).Inst());
            continue;
        default:
            FAIL("Unexpected constant-buffer pair bounds instruction");
        }
        values.emplace(&instruction, value);
    }
    return result;
}

IR::U32 Identity(Fixture& fixture, const IR::Value& value) {
    return IR::U32{IR::Value{
        &*fixture.block.PrependNewInst(fixture.block.end(), IR::Opcode::Identity, {value})}};
}

IR::Inst& Phi(Fixture& fixture) {
    auto& phi = *fixture.block.PrependNewInst(fixture.block.end(), IR::Opcode::Phi);
    phi.SetFlags(IR::Type::U32);
    return phi;
}

IR::U32 PairOffset(Fixture& fixture, u32 form) {
    auto& ir = fixture.ir;
    const auto input = ir.GetReg(IR::Reg::R0);
    const auto aligned = ir.BitwiseAnd(input, ir.Imm32(~7U));
    switch (form) {
    case 0:
        return aligned;
    case 1:
        return IR::U32{ir.ShiftLeftLogical(input, ir.Imm32(3U))};
    case 2:
        return IR::U32{ir.IAdd(Identity(fixture, Identity(fixture, aligned)), ir.Imm32(0xc0U))};
    case 3:
        return IR::U32{ir.ISub(aligned, ir.Imm32(8U))};
    case 4:
        return ir.BitwiseOr(aligned, ir.Imm32(8U));
    case 5:
        return ir.BitwiseXor(aligned, ir.Imm32(0xc0U));
    case 6:
        return ir.IMul(ir.BitwiseAnd(input, ir.Imm32(~1U)), ir.Imm32(4U));
    case 7:
        return IR::U32{
            ir.Select(ir.ILessThanEqual(input, ir.Imm32(7U), false), aligned, ir.Imm32(8U))};
    case 8:
        return Identity(fixture, Identity(fixture, aligned));
    case 9:
        return ir.BitwiseAnd(ir.Imm32(~7U), input);
    case 10: {
        const IR::U32 shifted{ir.ShiftLeftLogical(input, ir.Imm32(16U))};
        return ir.IMul(shifted, shifted);
    }
    default:
        return IR::U32{ir.IAdd(aligned, aligned)};
    }
}

u32 PairOffsetValue(u32 input, u32 form) {
    const u32 aligned = input & ~7U;
    switch (form) {
    case 1:
        return input << 3;
    case 2:
        return aligned + 0xc0U;
    case 3:
        return aligned - 8;
    case 4:
        return aligned | 8U;
    case 5:
        return aligned ^ 0xc0U;
    case 6:
        return (input & ~1U) * 4;
    case 7:
        return input <= 7 ? aligned : 8;
    case 10:
        return 0;
    case 11:
        return aligned + aligned;
    default:
        return aligned;
    }
}
} // namespace

TEST_CASE("Native constant-buffer scalar guards sanitize every actual load", "[native-shader]") {
    for (u32 bound : {64U, 1024U, 65536U}) {
        const std::array bounds{bound};
        for (bool floating : {false, true}) {
            for (bool dynamic : {false, true}) {
                for (u32 offset : {0U, 4U, bound - 4, bound - 3, bound - 1, bound, 0xffffffffU}) {
                    Fixture fixture;
                    const auto source =
                        dynamic ? fixture.ir.GetReg(IR::Reg::R0) : fixture.ir.Imm32(offset);
                    const IR::Value load =
                        floating ? IR::Value{fixture.ir.GetFloatCbuf(fixture.ir.Imm32(0U), source)}
                                 : IR::Value{fixture.ir.GetCbuf(fixture.ir.Imm32(0U), source)};
                    fixture.ir.Reference(load);
                    GuardConstantBufferLoads(fixture.program, bounds);
                    const u32 aligned = offset & ~3U;
                    const u32 expected = aligned <= bound - 4 ? 0x3f000000U + aligned : 0U;
                    REQUIRE(Evaluate(fixture.block, offset, bounds[0]) == expected);
                    if (dynamic || aligned > bound - 4) {
                        REQUIRE(load.Inst()->GetOpcode() == IR::Opcode::Identity);
                    }
                }
            }
        }
    }
}

TEST_CASE("Native constant-buffer bounds reject missing and unsupported bindings",
          "[native-shader]") {
    for (u32 size : {0U, 1U, 2U, 3U, 65U, 68U, 0x10040U}) {
        Fixture fixture;
        const auto value = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), fixture.ir.GetReg(IR::Reg::R0));
        fixture.ir.Reference(value);
        const std::array bounds{size};
        const auto count = fixture.block.size();
        REQUIRE_THROWS_AS(GuardConstantBufferLoads(fixture.program, bounds), Shader::RuntimeError);
        REQUIRE(fixture.block.size() == count);
        REQUIRE(value.Inst()->GetOpcode() == IR::Opcode::GetCbufU32);
    }
    for (bool dynamic_bank : {false, true}) {
        Fixture fixture;
        const auto bank = dynamic_bank ? fixture.ir.GetReg(IR::Reg::R0) : fixture.ir.Imm32(1U);
        fixture.ir.Reference(fixture.ir.GetCbuf(bank, fixture.ir.Imm32(0U)));
        constexpr std::array<u32, 1> bounds{64};
        REQUIRE_THROWS_AS(GuardConstantBufferLoads(fixture.program, bounds), Shader::RuntimeError);
    }
    Fixture fixture;
    fixture.ir.Reference(fixture.ir.GetCbuf(fixture.ir.Imm32(0U), fixture.ir.Imm32(0U)));
    REQUIRE_THROWS_AS(GuardConstantBufferLoads(fixture.program, {}), Shader::RuntimeError);
    Fixture invalid_bank;
    invalid_bank.ir.Reference(
        invalid_bank.ir.GetCbuf(invalid_bank.ir.Imm32(18U), invalid_bank.ir.Imm32(0U)));
    std::array<u32, 19> oversized_table;
    oversized_table.fill(64);
    REQUIRE_THROWS_AS(GuardConstantBufferLoads(invalid_bank.program, oversized_table),
                      Shader::RuntimeError);
}

TEST_CASE("Native static scalar loads keep proven small bounds and hardware alignment",
          "[native-shader]") {
    Fixture fixture;
    const auto value = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), fixture.ir.Imm32(3U));
    fixture.ir.Reference(value);
    constexpr std::array<u32, 1> bounds{4};
    const auto count = fixture.block.size();
    GuardConstantBufferLoads(fixture.program, bounds);
    REQUIRE(fixture.block.size() == count);
    REQUIRE(value.Inst()->GetOpcode() == IR::Opcode::GetCbufU32);
    REQUIRE(Evaluate(fixture.block, 0, 4) == 0x3f000000U);
    Fixture outside;
    outside.ir.Reference(outside.ir.GetCbuf(outside.ir.Imm32(0U), outside.ir.Imm32(4U)));
    REQUIRE_THROWS_AS(GuardConstantBufferLoads(outside.program, bounds), Shader::RuntimeError);
}

TEST_CASE("Native narrow and vector loads require a proven complete static packet",
          "[native-shader]") {
    constexpr std::array<u32, 1> bounds{64};
    for (u32 bits : {8U, 16U, 64U}) {
        for (bool signed_load : {false, true}) {
            Fixture fixture;
            const u32 width = bits / 8;
            const auto value = fixture.ir.GetCbuf(fixture.ir.Imm32(0U),
                                                  fixture.ir.Imm32(64 - width), bits, signed_load);
            fixture.ir.Reference(value);
            const auto opcode = value.Inst()->GetOpcode();
            const auto count = fixture.block.size();
            GuardConstantBufferLoads(fixture.program, bounds);
            REQUIRE(fixture.block.size() == count);
            REQUIRE(value.Inst()->GetOpcode() == opcode);
            for (bool dynamic : {false, true}) {
                Fixture unsupported;
                const auto offset =
                    dynamic ? unsupported.ir.GetReg(IR::Reg::R0) : unsupported.ir.Imm32(64U);
                unsupported.ir.Reference(
                    unsupported.ir.GetCbuf(unsupported.ir.Imm32(0U), offset, bits, signed_load));
                if (bits == 64) {
                    GuardConstantBufferLoads(unsupported.program, bounds);
                    REQUIRE(EvaluatePair(unsupported.block, 64, bounds[0]) ==
                            std::array<u32, 2>{0, 0});
                } else {
                    REQUIRE_THROWS_AS(GuardConstantBufferLoads(unsupported.program, bounds),
                                      Shader::RuntimeError);
                }
            }
        }
    }
    Fixture partial;
    partial.ir.Reference(partial.ir.GetCbuf(partial.ir.Imm32(0U), partial.ir.Imm32(0U), 64, false));
    constexpr std::array<u32, 1> short_bound{8};
    REQUIRE_THROWS_AS(GuardConstantBufferLoads(partial.program, short_bound), Shader::RuntimeError);
    for (u32 bits : {16U, 64U}) {
        Fixture unaligned;
        unaligned.ir.Reference(
            unaligned.ir.GetCbuf(unaligned.ir.Imm32(0U), unaligned.ir.Imm32(1U), bits, false));
        if (bits == 64) {
            GuardConstantBufferLoads(unaligned.program, bounds);
            REQUIRE(EvaluatePair(unaligned.block, 0, bounds[0]) == std::array<u32, 2>{0, 0});
        } else {
            REQUIRE_THROWS_AS(GuardConstantBufferLoads(unaligned.program, bounds),
                              Shader::RuntimeError);
        }
    }
}

TEST_CASE("Native constant-buffer guard validation precedes mutation across blocks",
          "[native-shader]") {
    Fixture fixture;
    const auto scalar = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), fixture.ir.GetReg(IR::Reg::R0));
    fixture.ir.Reference(scalar);
    IR::Block second{fixture.pool};
    fixture.program.blocks.push_back(&second);
    IR::IREmitter emitter{second};
    emitter.Reference(emitter.GetCbuf(emitter.Imm32(0U), emitter.GetReg(IR::Reg::R0), 16, false));
    const auto count = fixture.block.size();
    constexpr std::array<u32, 1> bounds{1024};
    REQUIRE_THROWS_AS(GuardConstantBufferLoads(fixture.program, bounds), Shader::RuntimeError);
    REQUIRE(fixture.block.size() == count);
    REQUIRE(scalar.Inst()->GetOpcode() == IR::Opcode::GetCbufU32);
}

TEST_CASE("Native aligned pair guards sanitize both words and every fallback packet",
          "[native-shader]") {
    for (u32 bound : {64U, 1024U, 65536U}) {
        const std::array bounds{bound};
        for (u32 form = 0; form < 12; ++form) {
            for (u32 input : {0U, 1U, 7U, 8U, bound - 8, bound, 0x80000000U, 0xffffffffU}) {
                CAPTURE(bound, form, input);
                Fixture fixture;
                const auto offset = PairOffset(fixture, form);
                const auto load = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), offset, 64, false);
                fixture.ir.Reference(load);
                GuardConstantBufferLoads(fixture.program, bounds);
                REQUIRE(load.Inst()->GetOpcode() == IR::Opcode::Identity);
                const u32 actual_offset = PairOffsetValue(input, form);
                REQUIRE(actual_offset % 8 == 0);
                const std::array expected =
                    actual_offset <= bound - 8
                        ? std::array{0x3f000000U + actual_offset, 0x40000000U + actual_offset + 4}
                        : std::array{0U, 0U};
                REQUIRE(EvaluatePair(fixture.block, input, bound) == expected);
            }
        }
        for (u32 offset : {0U, bound - 8, bound, 0xfffffff8U}) {
            Fixture fixture;
            const auto load =
                fixture.ir.GetCbuf(fixture.ir.Imm32(0U), fixture.ir.Imm32(offset), 64, false);
            fixture.ir.Reference(load);
            GuardConstantBufferLoads(fixture.program, bounds);
            const std::array expected =
                offset <= bound - 8 ? std::array{0x3f000000U + offset, 0x40000000U + offset + 4}
                                    : std::array{0U, 0U};
            REQUIRE(EvaluatePair(fixture.block, 0, bound) == expected);
            REQUIRE(load.Inst()->GetOpcode() ==
                    (offset <= bound - 8 ? IR::Opcode::GetCbufU32x2 : IR::Opcode::Identity));
        }
    }
}

TEST_CASE("Native pair guards check arbitrary addresses before every host packet",
          "[native-shader]") {
    for (u32 bound : {64U, 768U, 1024U, 65536U}) {
        const std::array bounds{bound};
        for (u32 form = 0; form < 4; ++form) {
            Fixture fixture;
            auto& ir = fixture.ir;
            IR::U32 offset = ir.GetReg(IR::Reg::R0);
            if (form == 1) {
                for (u32 index = 0; index < 80; ++index)
                    offset = Identity(fixture, offset);
            } else if (form == 2) {
                auto& phi = Phi(fixture);
                const IR::U32 carried{IR::Value{&phi}};
                const auto next = ir.IAdd(carried, ir.Imm32(8U));
                phi.AddPhiOperand(&fixture.block, ir.Imm32(0U));
                phi.AddPhiOperand(&fixture.block, next);
                offset = carried;
            } else if (form == 3) {
                offset = ir.BitCast<IR::U32>(ir.BitCast<IR::F32>(offset));
            }
            const auto load = ir.GetCbuf(ir.Imm32(0U), offset, 64, false);
            ir.Reference(load);
            GuardConstantBufferLoads(fixture.program, bounds);
            REQUIRE(load.Inst()->GetOpcode() == IR::Opcode::Identity);
            // Exhaust the low bits and binding edges, including lane 3 of a
            // vec4 packet, plus overflow and negative-looking addresses.
            for (u32 window : {0U, bound - 16, bound, 0xfffffff0U}) {
                for (u32 byte = 0; byte < 16; ++byte) {
                    const u32 address = window + byte;
                    const std::array expected =
                        !(address & 7) && address <= bound - 8
                            ? std::array{0x3f000000U + address, 0x40000000U + address + 4}
                            : std::array{0U, 0U};
                    REQUIRE(EvaluatePair(fixture.block, address, bound) == expected);
                }
            }
        }
    }
    for (u32 bound : {8U, 16U, 63U, 65U, 1023U, 65537U}) {
        Fixture fixture;
        const auto offset = fixture.ir.GetReg(IR::Reg::R0);
        const auto load = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), offset, 64, false);
        fixture.ir.Reference(load);
        const auto count = fixture.block.size();
        const std::array bounds{bound};
        REQUIRE_THROWS_AS(GuardConstantBufferLoads(fixture.program, bounds), Shader::RuntimeError);
        REQUIRE(fixture.block.size() == count);
        REQUIRE(load.Inst()->Arg(1) == offset);
    }
}

TEST_CASE("Native pair guards preserve arithmetic on loop-carried addresses", "[native-shader]") {
    for (u32 bound : {64U, 1024U, 65536U}) {
        const std::array bounds{bound};
        for (u32 form = 0; form < 5; ++form) {
            for (bool add_displacement : {false, true}) {
                for (u32 input : {0U, 1U, 7U, 8U, bound - 8, bound, 0x80000000U, 0xffffffffU}) {
                    CAPTURE(bound, form, add_displacement, input);
                    Fixture fixture;
                    auto& phi = Phi(fixture);
                    phi.AddPhiOperand(&fixture.block, fixture.ir.Imm32(0U));
                    phi.AddPhiOperand(&fixture.block, IR::Value{&phi});
                    const IR::U32 unknown{IR::Value{&phi}};
                    auto& ir = fixture.ir;
                    IR::U32 offset{ir.Imm32(0U)};
                    u32 actual_offset{};
                    switch (form) {
                    case 0:
                        offset = IR::U32{ir.ShiftLeftLogical(unknown, ir.Imm32(3U))};
                        actual_offset = input << 3;
                        break;
                    case 1:
                        offset = ir.BitwiseAnd(unknown, ir.Imm32(~7U));
                        actual_offset = input & ~7U;
                        break;
                    case 2:
                        offset = ir.BitwiseAnd(ir.Imm32(~7U), unknown);
                        actual_offset = input & ~7U;
                        break;
                    case 3:
                        offset = ir.IMul(unknown, ir.Imm32(8U));
                        actual_offset = input * 8;
                        break;
                    default:
                        offset = ir.IMul(ir.Imm32(8U), unknown);
                        actual_offset = input * 8;
                        break;
                    }
                    if (add_displacement) {
                        offset = IR::U32{
                            ir.IAdd(Identity(fixture, Identity(fixture, offset)), ir.Imm32(0xc0U))};
                        actual_offset += 0xc0U;
                    }
                    const auto pair = ir.GetCbuf(ir.Imm32(0U), offset, 64, false);
                    ir.Reference(pair);
                    GuardConstantBufferLoads(fixture.program, bounds);
                    REQUIRE(pair.Inst()->GetOpcode() == IR::Opcode::Identity);
                    const std::array expected = actual_offset <= bound - 8
                                                    ? std::array{0x3f000000U + actual_offset,
                                                                 0x40000000U + actual_offset + 4}
                                                    : std::array{0U, 0U};
                    REQUIRE(EvaluatePair(fixture.block, input, bound) == expected);
                }
            }
        }
    }
}

TEST_CASE("Native pair guards accept deep and shared offset graphs without expanding them",
          "[native-shader]") {
    constexpr std::array<u32, 1> bounds{1024};
    for (bool deep : {false, true}) {
        Fixture fixture;
        IR::U32 offset{fixture.ir.Imm32(0U)};
        if (deep) {
            offset = PairOffset(fixture, 0);
            for (u32 index = 0; index < 80; ++index) {
                offset = Identity(fixture, offset);
            }
        } else {
            const auto tree = [&](auto&& self, u32 depth) -> IR::U32 {
                if (!depth) {
                    return fixture.ir.BitwiseAnd(fixture.ir.GetReg(IR::Reg::R0),
                                                 fixture.ir.Imm32(~7U));
                }
                return fixture.ir.BitwiseOr(self(self, depth - 1), self(self, depth - 1));
            };
            offset = tree(tree, 9);
        }
        const auto load = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), offset, 64, false);
        fixture.ir.Reference(load);
        const auto count = fixture.block.size();
        GuardConstantBufferLoads(fixture.program, bounds);
        REQUIRE(fixture.block.size() <= count + 16);
        REQUIRE(load.Inst()->GetOpcode() == IR::Opcode::Identity);
        REQUIRE(EvaluatePair(fixture.block, 8, bounds[0]) ==
                std::array<u32, 2>{0x3f000008, 0x4000000c});
    }
}

TEST_CASE("Native pair guard validation is atomic when a later load is unsupported",
          "[native-shader]") {
    Fixture fixture;
    const auto offset = PairOffset(fixture, 2);
    const auto pair = fixture.ir.GetCbuf(fixture.ir.Imm32(0U), offset, 64, false);
    fixture.ir.Reference(pair);
    IR::Block second{fixture.pool};
    fixture.program.blocks.push_back(&second);
    IR::IREmitter emitter{second};
    emitter.Reference(emitter.GetCbuf(emitter.Imm32(0U), emitter.GetReg(IR::Reg::R0), 16, false));
    const auto count = fixture.block.size();
    constexpr std::array<u32, 1> bounds{1024};
    REQUIRE_THROWS_AS(GuardConstantBufferLoads(fixture.program, bounds), Shader::RuntimeError);
    REQUIRE(fixture.block.size() == count);
    REQUIRE(pair.Inst()->GetOpcode() == IR::Opcode::GetCbufU32x2);
}
