// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <span>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "video_core/control/native_compute.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_macro_processor.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/turing_macro.h"

namespace {
using Engine = Tegra::TuringMacroEngine;
using Maxwell3D = Tegra::Engines::Maxwell3D;
using Inst = Engine::Instruction;
using Op = Engine::Operation;
constexpr auto Z = Engine::Zero;
constexpr auto I = Engine::Immediate;
constexpr auto L0 = Engine::Load0;
constexpr auto L1 = Engine::Load1;

Engine::EncodedInstruction Encode(const Inst& inst) {
    std::array<u32, 3> bits{};
    const auto put = [&](u32 first, u32 count, u32 value) {
        for (u32 bit = 0; bit < count; ++bit) {
            bits[(first + bit) / 32] |= ((value >> bit) & 1) << ((first + bit) % 32);
        }
    };
    put(0, 1, inst.end_next);
    put(1, 4, inst.predicate_mode);
    put(5, 5, inst.predicate);
    for (u32 slot = 0; slot < 2; ++slot) {
        const u32 first = slot == 0 ? 10 : 46;
        put(first, 5, static_cast<u8>(inst.alu[slot].operation));
        put(first + 5, 5, inst.alu[slot].destination);
        put(first + 10, 5, inst.alu[slot].source_a);
        put(first + 15, 5, inst.alu[slot].source_b);
        put(first + 20, 16, inst.immediate[slot]);
        put(82 + slot * 7, 3, inst.output[slot].method);
        put(85 + slot * 7, 4, inst.output[slot].emit);
    }
    return {bits[2], bits[1], bits[0]};
}

void Upload(Engine& engine, std::span<const Inst> program, u32 start = 0) {
    REQUIRE(engine.SetInstructionPointer(start));
    for (const auto& inst : program) {
        for (const auto word : Encode(inst)) {
            REQUIRE(engine.UploadInstruction(word));
        }
    }
}

Inst Method() {
    Inst inst;
    inst.immediate[0] = 0x1040;
    inst.output[0].method = 5;
    return inst;
}

struct Host {
    std::vector<u32> parameters;
    size_t cursor{};
    std::vector<std::pair<u16, u32>> methods;
    u32 state_value{0x12345678};
    u16 state_address{};
    u32 reads{};
    u32 barriers{};
    bool fail_read{};
    size_t emit_limit{10000};
    bool fail_barrier{};
    bool cancelled{};
    bool replace_parameters{};

    Engine::Host Bind() {
        return {
            this,
            [](void* context, u32& value) {
                auto& self = *static_cast<Host*>(context);
                if (self.cursor == self.parameters.size())
                    return false;
                value = self.parameters[self.cursor++];
                return true;
            },
            [](void* context, u16 address, u32& value) {
                auto& self = *static_cast<Host*>(context);
                ++self.reads;
                self.state_address = address;
                value = self.state_value;
                return !self.fail_read;
            },
            [](void* context, u16 address, u32 value) {
                auto& self = *static_cast<Host*>(context);
                if (self.methods.size() == self.emit_limit)
                    return false;
                self.methods.emplace_back(address, value);
                return true;
            },
            [](void* context) {
                auto& self = *static_cast<Host*>(context);
                ++self.barriers;
                if (self.replace_parameters) {
                    self.parameters = {0x2222, 0x3333};
                    self.cursor = 0;
                }
                return !self.fail_barrier;
            },
            [](void* context) { return static_cast<Host*>(context)->cancelled; },
        };
    }
};
} // namespace

TEST_CASE("Turing MME decodes reversed instruction words and bounds its uploaded RAM",
          "[video_core][ipc]") {
    // An independently packed instruction exercises both immediate and source fields crossing
    // word boundaries, plus the high-word output selectors.
    const auto decoded = Engine::Decode({0xa742af34, 0x59a367f9, 0xbbe3876b});
    REQUIRE(decoded.end_next);
    REQUIRE(decoded.predicate_mode == 5);
    REQUIRE(decoded.predicate == 27);
    REQUIRE(static_cast<u8>(decoded.alu[0].operation) == 1);
    REQUIRE(decoded.alu[0].destination == 7);
    REQUIRE(decoded.alu[0].source_a == 30);
    REQUIRE(decoded.alu[0].source_b == 29);
    REQUIRE(decoded.immediate[0] == 0x9fe6);
    REQUIRE(static_cast<u8>(decoded.alu[1].operation) == 13);
    REQUIRE(decoded.alu[1].destination == 20);
    REQUIRE(decoded.alu[1].source_a == 25);
    REQUIRE(decoded.alu[1].source_b == 2);
    REQUIRE(decoded.immediate[1] == 0xabcd);
    REQUIRE(decoded.output[0].method == 0);
    REQUIRE(decoded.output[0].emit == 10);
    REQUIRE(decoded.output[1].method == 3);
    REQUIRE(decoded.output[1].emit == 10);

    Engine engine;
    Inst stop;
    stop.alu[0].operation = Op::BranchEqual;
    stop.immediate[0] = 0x9000;
    const auto words = Encode(stop);
    REQUIRE(engine.Execute(0, {}, 10).error == Engine::Error::IncompleteInstruction);
    REQUIRE(engine.SetInstructionPointer(Engine::InstructionCount - 1));
    REQUIRE(engine.UploadInstruction(words[0]));
    REQUIRE(engine.UploadInstruction(words[1]));
    REQUIRE(engine.Execute(Engine::InstructionCount - 1, {}, 10).error ==
            Engine::Error::IncompleteInstruction);
    REQUIRE(engine.UploadInstruction(words[2]));
    REQUIRE_FALSE(engine.UploadInstruction(0xffffffff));
    REQUIRE_FALSE(engine.SetInstructionPointer(Engine::InstructionCount));
    const auto result = engine.Execute(Engine::InstructionCount - 1, {}, 10);
    REQUIRE(result.Succeeded());
    REQUIRE(result.instructions == 1);
    REQUIRE(engine.Execute(Engine::InstructionCount, {}, 10).error ==
            Engine::Error::InvalidProgramCounter);
    Upload(engine, std::array{stop}, 7);
    REQUIRE(engine.Execute(7, {}, 10).Succeeded());
}

TEST_CASE("Turing MME executes dual ALUs with forwarding and signed arithmetic",
          "[video_core][ipc]") {
    struct Case {
        Op op;
        u32 x;
        u32 y;
        u32 expected;
        u16 immediate{};
    };
    const std::array cases{
        Case{Op::Add, 0xffffffff, 1, 0},
        Case{Op::Subtract, 0, 1, 0xffffffff},
        Case{Op::Multiply, 0xfffffffe, 3, 0xfffffffa},
        Case{Op::MultiplyUnsigned, 0xfffffffe, 3, 0xfffffffa},
        Case{Op::CountLeadingZeros, 0, 0, 32},
        Case{Op::CountLeadingZeros, 0x8000, 0, 16},
        Case{Op::ShiftLeft, 3, 35, 24},
        Case{Op::ShiftRight, 0x80000000, 33, 0x40000000},
        Case{Op::ShiftRightSigned, 0x80000000, 33, 0xc0000000},
        Case{Op::And, 0xff00, 0x1234, 0x1200},
        Case{Op::Nand, 0xff00, 0x1234, 0xffffedff},
        Case{Op::Or, 0xff00, 0x1234, 0xff34},
        Case{Op::Xor, 0xff00, 0x1234, 0xed34},
        Case{Op::Merge, 0xffff0000, 0xabcdef12, 0xfffff100, (8 << 10) | (8 << 5) | 4},
        Case{Op::Less, 0xffffffff, 0, 0xffffffff},
        Case{Op::LessUnsigned, 0xffffffff, 0, 0},
        Case{Op::LessEqual, 0xffffffff, 0, 0xffffffff},
        Case{Op::LessEqualUnsigned, 0xffffffff, 0, 0},
        Case{Op::Equal, 7, 7, 0xffffffff},
    };
    Engine engine;
    for (const auto& test : cases) {
        Inst work;
        work.alu[0] = {test.op, 23, L0, L1};
        work.alu[1] = {Op::Add, Z, 23, Z};
        work.immediate[0] = test.immediate;
        work.output = {Engine::Output{0, 1}, Engine::Output{0, 2}};
        work.end_next = true;
        Upload(engine, std::array{Method(), work, Inst{}});
        Host host;
        host.parameters = {test.x, test.y};
        const auto result = engine.Execute(0, host.Bind(), 20);
        REQUIRE(result.Succeeded());
        REQUIRE(result.instructions == 3);
        REQUIRE(result.parameters == 2);
        REQUIRE(result.methods == 2);
        REQUIRE(host.methods ==
                std::vector<std::pair<u16, u32>>{{0x100, test.expected}, {0x104, test.expected}});
    }
    for (const auto op : {Op::AddCarry, Op::SubtractBorrow, Op::MultiplyHigh}) {
        Inst work;
        work.alu[0] = {op == Op::AddCarry         ? Op::Add
                       : op == Op::SubtractBorrow ? Op::Subtract
                                                  : Op::Multiply,
                       Z, L0, L1};
        work.alu[1] = {op, Z, op == Op::MultiplyHigh ? Z : I, Z};
        work.immediate[1] = 10;
        work.end_next = true;
        work.output = {Engine::Output{0, 1}, Engine::Output{0, 2}};
        Upload(engine, std::array{Method(), work, Inst{}});
        Host host;
        host.parameters = op == Op::SubtractBorrow ? std::vector<u32>{0, 1}
                          : op == Op::MultiplyHigh ? std::vector<u32>{0xfffffffe, 3}
                                                   : std::vector<u32>{0xffffffff, 1};
        REQUIRE(engine.Execute(0, host.Bind(), 20).Succeeded());
        REQUIRE(host.methods[1].second == (op == Op::AddCarry         ? 11U
                                           : op == Op::SubtractBorrow ? 9U
                                                                      : 0xffffffffU));
    }
}

TEST_CASE("Turing MME keeps register 23 separate from its ALU result latches",
          "[video_core][ipc]") {
    Engine engine;
    Inst save, forward;
    save.alu[0] = {Op::Add, 23, L0, Z};
    forward.alu[0] = {Op::Add, Z, I, Z};
    forward.immediate[0] = 0x33;
    forward.alu[1] = {Op::Add, Z, 23, Z};
    forward.output[0].emit = 2;
    forward.end_next = true;
    Upload(engine, std::array{Method(), save, forward, Inst{}});
    Host host;
    host.parameters = {0x11112222};
    REQUIRE(engine.Execute(0, host.Bind(), 20).Succeeded());
    REQUIRE(host.methods.back().second == 0x11112222);
}

TEST_CASE("Turing MME predication consumes loads before enabling individual execution slots",
          "[video_core][ipc]") {
    constexpr std::array<u8, 16> when_false{15, 15, 0,  15, 12, 13, 15, 14,
                                            15, 7,  15, 11, 3,  11, 15, 7};
    constexpr std::array<u8, 16> when_true{15, 0,  15, 12, 15, 14, 14, 15,
                                           3,  11, 11, 7,  15, 15, 7,  15};
    Engine engine;
    for (u8 mode = 0; mode < 16; ++mode) {
        for (u32 predicate = 0; predicate < 2; ++predicate) {
            auto setup = Method();
            setup.alu[0] = {Op::Add, 0, L0, Z};
            Inst work;
            work.predicate_mode = mode;
            work.predicate = 0;
            work.alu[0] = {Op::State, Z, L0, Z};
            work.alu[1] = {Op::DataWrite, Z, L0, L1};
            work.output = {Engine::Output{0, 3}, Engine::Output{0, 4}};
            work.end_next = true;
            Upload(engine, std::array{setup, work, Inst{}});
            REQUIRE(engine.WriteData(7, 0x7777));
            Host host;
            host.parameters = {predicate, 7, 0x9876};
            const auto result = engine.Execute(0, host.Bind(), 20);
            REQUIRE(result.Succeeded());
            REQUIRE(result.parameters == 3);
            REQUIRE(host.cursor == 3);
            const u8 mask = predicate ? when_true[mode] : when_false[mode];
            REQUIRE(host.reads == ((mask & 1) != 0 ? 1U : 0U));
            u32 data{};
            REQUIRE(engine.ReadData(7, data));
            REQUIRE(data == ((mask & 2) != 0 ? 0x9876U : 0x7777U));
            std::vector<std::pair<u16, u32>> expected;
            if ((mask & 4) != 0)
                expected.emplace_back(0x100, 7);
            if ((mask & 8) != 0)
                expected.emplace_back(expected.empty() ? 0x100 : 0x104, 0x9876);
            REQUIRE(host.methods == expected);
        }
    }
}

TEST_CASE("Turing MME taken branch bit14 suppresses only its own end marker", "[video_core][ipc]") {
    for (const bool cancel_end : {false, true}) {
        Engine engine;
        Inst initialize, body, decrement, branch, delay;
        initialize.alu[0] = {Op::Add, 0, I, Z};
        initialize.immediate[0] = 3;
        body.alu[0] = {Op::Add, Z, 0, Z};
        body.output[0].emit = 1;
        decrement.alu[0] = {Op::Subtract, 0, 0, I};
        decrement.immediate[0] = 1;
        branch.alu[0] = {Op::BranchEqual, Z, 0, Z};
        branch.immediate[0] = cancel_end ? 0x5ffe : 0x1ffe;
        branch.end_next = true;
        delay.immediate[0] = 0xbeef;
        delay.output[0].emit = 5;
        Upload(engine, std::array{Method(), initialize, body, decrement, branch, delay});
        Host host;
        const auto result = engine.Execute(0, host.Bind(), 100);
        REQUIRE(result.Succeeded());
        if (cancel_end) {
            REQUIRE(host.methods == std::vector<std::pair<u16, u32>>{
                                        {0x100, 3}, {0x104, 2}, {0x108, 1}, {0x10c, 0xbeef}});
            REQUIRE(result.instructions == 12);
        } else {
            REQUIRE(host.methods == std::vector<std::pair<u16, u32>>{{0x100, 3}, {0x104, 2}});
            REQUIRE(result.instructions == 6);
        }
    }
    Engine engine;
    Inst arm, branch, unreachable;
    arm.end_next = true;
    branch.alu[0] = {Op::BranchEqual, Z, Z, Z};
    branch.immediate[0] = 0xc001;
    branch.end_next = true;
    unreachable.immediate[0] = 0xbeef;
    unreachable.output[0].emit = 5;
    Upload(engine, std::array{Method(), arm, branch, unreachable});
    Host host;
    const auto result = engine.Execute(0, host.Bind(), 100);
    REQUIRE(result.Succeeded());
    REQUIRE(result.instructions == 3);
    REQUIRE(host.methods.empty());
}

TEST_CASE("Turing MME follows counted loops, signed branches and exit delay slots",
          "[video_core][ipc]") {
    Engine engine;
    Inst initialize, decrement, branch, output, delay;
    initialize.alu[0] = {Op::Add, 0, I, Z};
    initialize.immediate[0] = 3;
    decrement.alu[0] = {Op::Subtract, 0, 0, I};
    decrement.immediate[0] = 1;
    branch.alu[0] = {Op::BranchEqual, Z, 0, Z};
    branch.immediate[0] = 0x1fff;
    output.alu[0] = {Op::Add, Z, 0, Z};
    output.output[0].emit = 1;
    output.end_next = true;
    delay.immediate[0] = 0xbeef;
    delay.output[0].emit = 5;
    Upload(engine, std::array{Method(), initialize, decrement, branch, output, delay});
    Host host;
    const auto result = engine.Execute(0, host.Bind(), 100);
    REQUIRE(result.Succeeded());
    REQUIRE(result.instructions == 10);
    REQUIRE(host.methods == std::vector<std::pair<u16, u32>>{{0x100, 0}, {0x104, 0xbeef}});

    Inst loop, body0, body1, finish;
    loop.alu[0] = {Op::Loop, Z, L0, Z};
    loop.immediate[0] = 3;
    body0.immediate[0] = 7;
    body0.output[0].emit = 5;
    body1.immediate[0] = 8;
    body1.output[0].emit = 5;
    finish.end_next = true;
    Upload(engine, std::array{Method(), loop, body0, body1, finish, Inst{}});
    for (const u32 count : {0U, 1U, 3U}) {
        Host counted;
        counted.parameters = {count};
        const auto completed = engine.Execute(0, counted.Bind(), 100);
        REQUIRE(completed.Succeeded());
        const u32 iterations = count;
        REQUIRE(completed.instructions == iterations * 2 + 4);
        REQUIRE(counted.methods.size() == iterations * 2);
        if (count != 0) {
            REQUIRE(counted.methods.front().second == 7);
            REQUIRE(counted.methods.back().second == 8);
        }
    }
    Inst consume;
    consume.alu[0] = {Op::Add, Z, L0, L1};
    consume.output[0].emit = 1;
    loop.immediate[0] = 2;
    Upload(engine, std::array{Method(), loop, consume, finish, Inst{}});
    Host empty;
    empty.parameters = {0};
    const auto skipped = engine.Execute(0, empty.Bind(), 20);
    REQUIRE(skipped.Succeeded());
    REQUIRE(skipped.parameters == 1);
    REQUIRE(skipped.instructions == 4);
    REQUIRE(empty.methods.empty());
    Host loaded;
    loaded.parameters = {2, 11, 12, 13, 14};
    const auto consumed = engine.Execute(0, loaded.Bind(), 20);
    REQUIRE(consumed.Succeeded());
    REQUIRE(consumed.parameters == 5);
    REQUIRE(loaded.methods == std::vector<std::pair<u16, u32>>{{0x100, 23}, {0x104, 27}});
    Inst jump, unreachable, exit;
    jump.alu[0].operation = Op::Jump;
    jump.immediate[0] = 0x8002;
    unreachable.alu[0].source_a = L0;
    exit.alu[0].operation = Op::BranchEqual;
    exit.immediate[0] = 0x9000;
    Upload(engine, std::array{Method(), jump, unreachable, exit});
    Host jumped;
    REQUIRE(engine.Execute(0, jumped.Bind(), 10).Succeeded());
    jump.immediate[0] = 0x8000;
    Upload(engine, std::array{jump});
    REQUIRE(engine.Execute(0, {}, 17).error == Engine::Error::InstructionLimit);
    branch.alu[0] = {Op::BranchEqual, Z, Z, Z};
    branch.immediate[0] = 0x9fff;
    Upload(engine, std::array{branch});
    REQUIRE(engine.Execute(0, {}, 10).error == Engine::Error::InvalidBranch);
}

TEST_CASE("Turing MME retains data RAM and propagates host failures without later emissions",
          "[video_core][ipc]") {
    Engine engine;
    Inst write, read;
    write.alu[0] = {Op::DataWrite, Z, I, L0};
    write.immediate[0] = Engine::DataCount - 1;
    read.alu[0] = {Op::DataRead, 23, I, Z};
    read.immediate[0] = Engine::DataCount - 1;
    read.end_next = true;
    read.output[0].emit = 1;
    Upload(engine, std::array{Method(), write, read, Inst{}});
    Host host;
    host.parameters = {0x87654321};
    REQUIRE(engine.Execute(0, host.Bind(), 20).Succeeded());
    REQUIRE(host.methods.back().second == 0x87654321);
    Host again;
    REQUIRE(engine.Execute(2, again.Bind(), 20).Succeeded());
    REQUIRE(again.methods.back().second == 0x87654321);
    u32 untouched{0x99};
    REQUIRE_FALSE(engine.ReadData(Engine::DataCount, untouched));
    REQUIRE(untouched == 0x99);
    REQUIRE_FALSE(engine.WriteData(Engine::DataCount, 1));
    read.immediate[0] = Engine::DataCount;
    Upload(engine, std::array{read, Inst{}});
    REQUIRE(engine.Execute(0, {}, 20).error == Engine::Error::InvalidDataAddress);

    Inst barrier, consume;
    barrier.alu[0] = {Op::Extended, Z, I, Engine::PairedImmediate};
    barrier.immediate = {0x1000, 1};
    consume.alu[0] = {Op::Add, Z, L0, L1};
    consume.end_next = true;
    consume.output[0].emit = 1;
    Upload(engine, std::array{Method(), barrier, consume, Inst{}});
    Host fifo;
    fifo.replace_parameters = true;
    REQUIRE(engine.Execute(0, fifo.Bind(), 20).Succeeded());
    REQUIRE(fifo.barriers == 1);
    REQUIRE(fifo.methods.back().second == 0x5555);
    Host failed_barrier;
    failed_barrier.fail_barrier = true;
    REQUIRE(engine.Execute(0, failed_barrier.Bind(), 20).error == Engine::Error::BarrierFailed);
    REQUIRE(failed_barrier.methods.empty());
    Host underflow;
    REQUIRE(engine.Execute(2, underflow.Bind(), 20).error == Engine::Error::ParameterUnderflow);
    REQUIRE(underflow.methods.empty());

    Inst state;
    state.alu[0] = {Op::State, Z, I, Z};
    state.immediate[0] = 0x123;
    state.output = {Engine::Output{0, 1}, Engine::Output{0, 1}};
    state.end_next = true;
    Upload(engine, std::array{Method(), state, Inst{}});
    Host fault;
    fault.fail_read = true;
    REQUIRE(engine.Execute(0, fault.Bind(), 20).error == Engine::Error::StateReadFailed);
    REQUIRE(fault.state_address == 0x48c);
    REQUIRE(fault.methods.empty());
    Host reject;
    reject.emit_limit = 1;
    const auto partial = engine.Execute(0, reject.Bind(), 20);
    REQUIRE(partial.error == Engine::Error::MethodEmitFailed);
    REQUIRE(partial.methods == 1);
    REQUIRE(reject.methods.size() == 1);
    Host cancelled;
    cancelled.cancelled = true;
    REQUIRE(engine.Execute(0, cancelled.Bind(), 20).error == Engine::Error::Cancelled);
    Inst undefined;
    undefined.alu[0].source_a = 23;
    Upload(engine, std::array{undefined});
    REQUIRE(engine.Execute(0, {}, 20).error == Engine::Error::UndefinedRegister);
    undefined.alu[0].source_a = L1;
    Upload(engine, std::array{undefined});
    REQUIRE(engine.Execute(0, {}, 20).error == Engine::Error::InvalidInstruction);
}

namespace {
using Processor = Tegra::Control::NativeMacroProcessor;
struct MethodHost {
    std::array<u32, 0xe00> registers{};
    std::vector<std::pair<u32, u32>> writes;
    bool reject_write{};
    bool cancelled{};
    Processor::Host Bind() {
        return {this,
                [](void* context, u32 method, u32& value) {
                    value = static_cast<MethodHost*>(context)->registers.at(method);
                    return true;
                },
                [](void* context, u32 method, u32 value) {
                    auto& self = *static_cast<MethodHost*>(context);
                    if (self.reject_write)
                        return false;
                    self.registers.at(method) = value;
                    self.writes.emplace_back(method, value);
                    return true;
                },
                nullptr,
                [](void* context) { return static_cast<MethodHost*>(context)->cancelled; }};
    }
};
std::array<Inst, 3> SumProgram(u32 method) {
    auto set_method = Method();
    set_method.immediate[0] = static_cast<u16>(0x1000 | method);
    Inst sum;
    sum.alu[0] = {Op::Add, 0, L0, L1};
    sum.output[0].emit = 1;
    sum.end_next = true;
    return {set_method, sum, Inst{}};
}
void Upload(Processor& processor, std::span<const Inst> program, MethodHost& host, u32 start = 7,
            u32 slot = 5) {
    REQUIRE(processor.CallMethod(0x45, start, true, host.Bind()) == Processor::Status::Accepted);
    for (const auto& instruction : program) {
        for (u32 word : Encode(instruction)) {
            REQUIRE(processor.CallMethod(0x46, word, true, host.Bind()) ==
                    Processor::Status::Accepted);
        }
    }
    REQUIRE(processor.CallMethod(0x47, slot, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(processor.CallMethod(0x48, start, true, host.Bind()) == Processor::Status::Accepted);
}
} // namespace

TEST_CASE("Native macro methods retain parameters across command fragments and share uploads",
          "[ipc][video_core]") {
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    Processor first{state}, second{state};
    MethodHost host;
    Upload(first, SumProgram(0xd00), host);
    REQUIRE(first.CallMethod(0xe0a, 8, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(first.HasPendingInvocation());
    REQUIRE(second.CallMethod(0xe0a, 2, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(first.CallMethod(0xe0b, 9, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE_FALSE(first.HasPendingInvocation());
    REQUIRE(second.CallMethod(0xe0b, 3, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(host.writes == std::vector<std::pair<u32, u32>>{{0xd00, 17}, {0xd00, 5}});
    REQUIRE(first.GetExecutionResult().parameters == 2);
    REQUIRE(first.GetExecutionResult().instructions == 3);
    REQUIRE(first.CallMethod(0x202, 99, true, host.Bind()) == Processor::Status::Unhandled);
    REQUIRE(host.registers[0x202] == 0);
    host.reject_write = true;
    REQUIRE(first.CallMethod(0xe0a, 1, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(first.CallMethod(0xe0b, 2, true, host.Bind()) == Processor::Status::Failed);
    REQUIRE(first.GetExecutionResult().error == Engine::Error::MethodEmitFailed);
    host.reject_write = false;
    REQUIRE(first.CallMethod(0x45, 0, true, host.Bind()) == Processor::Status::Failed);
    REQUIRE(host.writes.size() == 2);
}

TEST_CASE("Native macro methods reject unbound calls, bad RAM pointers and interleaved writes",
          "[ipc][video_core]") {
    MethodHost host;
    for (const auto [method, value] :
         std::array<std::pair<u32, u32>, 4>{{{0x45, static_cast<u32>(Engine::InstructionCount)},
                                             {0x47, 0x80},
                                             {0x48, static_cast<u32>(Engine::InstructionCount)},
                                             {0xe00, 1}}}) {
        auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
        Processor processor{state};
        REQUIRE(processor.CallMethod(method, value, true, host.Bind()) ==
                Processor::Status::Failed);
        REQUIRE(processor.CallMethod(0x45, 0, true, host.Bind()) == Processor::Status::Failed);
    }
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    Processor odd{state};
    Upload(odd, SumProgram(0xd00), host);
    REQUIRE(odd.CallMethod(0xe0b, 0, true, host.Bind()) == Processor::Status::Failed);
    Processor interleaved{state};
    REQUIRE(interleaved.CallMethod(0xe0a, 1, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(interleaved.CallMethod(0xd00, 2, true, host.Bind()) == Processor::Status::Failed);
    REQUIRE(host.writes.empty());
    Processor bounded{state};
    REQUIRE(bounded.CallMethod(0xe0a, 0, false, host.Bind()) == Processor::Status::Accepted);
    for (size_t index = 1; index < Processor::MaxParameters; ++index)
        REQUIRE(bounded.CallMethod(0xe0b, 0, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(bounded.CallMethod(0xe0b, 0, true, host.Bind()) == Processor::Status::Failed);
    REQUIRE(bounded.GetError() == Processor::Error::ParameterLimit);
    Processor cancelled{state};
    host.cancelled = true;
    REQUIRE(cancelled.CallMethod(0x45, 0, true, host.Bind()) == Processor::Status::Failed);
    REQUIRE(cancelled.GetError() == Processor::Error::Cancelled);
}

TEST_CASE("The graphics engine executes native macros without using legacy macro code",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    Tegra::Engines::Maxwell3D engine{space};
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    engine.SetNativeMacroState(state);
    engine.CallMethod(system, 0x45, 7, true);
    std::vector<u32> words;
    for (const auto& instruction : SumProgram(0xd00))
        for (u32 word : Encode(instruction))
            words.push_back(word);
    engine.CallMultiMethod(system, 0x46, words.data(), static_cast<u32>(words.size()),
                           static_cast<u32>(words.size()));
    engine.CallMethod(system, 0x47, 5, true);
    engine.CallMethod(system, 0x48, 7, true);
    engine.CallMethod(system, MAXWELL3D_REG_INDEX(shadow_ram_control), 1, true);
    REQUIRE_FALSE(engine.HasNativeMacroFailure());
    engine.CallMethod(system, 0xe0a, 8, false);
    const u32 last = 9;
    engine.CallMultiMethod(system, 0xe0b, &last, 1, 1);
    REQUIRE_FALSE(engine.HasNativeMacroFailure());
    REQUIRE(engine.GetRegisterValue(0xd00) == 17);
    engine.CallMethod(system, 0xe1f, 1, true);
    REQUIRE(engine.HasNativeMacroFailure());
    engine.CallMethod(system, 0xd00, 99, true);
    REQUIRE(engine.GetRegisterValue(0xd00) == 17);
}

TEST_CASE("Native graphics configuration survives channel binding in the same group",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    Tegra::Engines::Maxwell3D first{space};
    Tegra::Engines::Maxwell3D second{space};
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    first.SetNativeMacroState(state);
    const Tegra::Control::NativeGraphicsConfig::Host host{nullptr, [](void*) { return true; },
                                                          nullptr};
    REQUIRE(state->graphics_config.ApplyMaskedRegister(0x418800, 1, 1, host) ==
            Tegra::Control::NativeGraphicsConfig::Error::None);
    second.SetNativeMacroState(state);
    REQUIRE(first.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
    REQUIRE(second.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
    REQUIRE(&first.GetNativeGraphicsConfig() == &second.GetNativeGraphicsConfig());
    first.SetNativeMacroState(state);
    REQUIRE(first.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
    first.SetNativeMacroState(nullptr);
    REQUIRE_FALSE(first.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
    REQUIRE(second.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
    second.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
    REQUIRE_FALSE(second.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
    REQUIRE(state->graphics_config.PolygonOffsetNanIsZero());
}

TEST_CASE("Native constant-buffer bindings retain canonical addresses without physical cache binds",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    space.BindRasterizer(&rasterizer);
    constexpr GPUVAddr base = 0x8000000;
    REQUIRE(space.MapSparse(base, 0x1000, false) == base);
    Tegra::Engines::Maxwell3D engine{space};
    engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
    REQUIRE(engine.CallNativeMethod(system, 0x8e0, 32, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x8e1, 0, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x8e2, base + 0x100, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x8e3, 0xffffffff, true, {}));
    for (u32 stage = 0; stage < 5; ++stage) {
        REQUIRE(engine.CallNativeMethod(system, 0x904 + stage * 8, 0x111, true, {}));
        const auto& binding = engine.state.shader_stages[stage].const_buffers[17];
        REQUIRE(binding.enabled);
        REQUIRE(binding.address == base + 0x100);
        REQUIRE(binding.size == 32);
    }
    REQUIRE(engine.CallNativeMethod(system, 0x8e0, 0xffffffff, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x8e1, 0xffffffff, true, {}));
    REQUIRE(engine.CallNativeMethod(system, 0x904, 0x110, true, {}));
    REQUIRE_FALSE(engine.state.shader_stages[0].const_buffers[17].enabled);
    REQUIRE_FALSE(engine.CallNativeMethod(system, 0x904, 0x120, true, {}));
    REQUIRE(engine.HasNativeMacroFailure());
}

TEST_CASE("Native constant-buffer bindings reject bad fields and incomplete readable ranges",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    space.BindRasterizer(&rasterizer);
    constexpr GPUVAddr base = 0x8000000;
    REQUIRE(space.MapSparse(base, 0x1000, false) == base);
    for (u32 invalid : {0U, 1U, 2U, 3U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        REQUIRE(engine.CallNativeMethod(system, 0x8e0, 32, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x8e1, invalid == 0 ? 0x100 : 0, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x8e2, base + 0xff0, true, {}));
        const u32 raw = invalid == 2 ? 0x121 : (invalid == 3 ? 3 : 1);
        REQUIRE_FALSE(engine.CallNativeMethod(system, 0x904, raw, true, {}));
        REQUIRE_FALSE(engine.state.shader_stages[0].const_buffers[0].enabled);
    }
}

TEST_CASE("Native instrumentation initialization retains only its complete metadata pair",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    Tegra::Engines::Maxwell3D engine{space};
    engine.SetNativeMacroState(state);
    REQUIRE_FALSE(engine.HasPendingNativeInstrumentation());
    REQUIRE(engine.CallNativeMethod(system, 0x54, 0x49000000, true, {}));
    REQUIRE(engine.HasPendingNativeInstrumentation());
    REQUIRE(engine.GetRegisterValue(0x54) == 0x49000000);
    REQUIRE(engine.CallNativeMethod(system, 0x55, 0x49000001, true, {}));
    REQUIRE_FALSE(engine.HasPendingNativeInstrumentation());
    REQUIRE(engine.GetRegisterValue(0x55) == 0x49000001);
    REQUIRE_FALSE(engine.CallNativeMethod(system, 0x55, 0x49000001, true, {}));
    REQUIRE(engine.HasNativeMacroFailure());
    Tegra::Engines::Maxwell3D nested{space};
    nested.SetNativeMacroState(state);
    REQUIRE(nested.CallNativeMethod(system, 0x54, 0x49000000, true, {}));
    REQUIRE_FALSE(nested.CallNativeMethod(system, 0x54, 0x49000000, true, {}));
    REQUIRE(nested.HasPendingNativeInstrumentation());
    Tegra::Engines::Maxwell3D wrong_data{space};
    wrong_data.SetNativeMacroState(state);
    REQUIRE(wrong_data.CallNativeMethod(system, 0x54, 0x49000000, true, {}));
    REQUIRE_FALSE(wrong_data.CallNativeMethod(system, 0x55, 0, true, {}));
    REQUIRE(wrong_data.HasPendingNativeInstrumentation());
}

TEST_CASE("Native constant-buffer failure preserves the upload cursor", "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    for (u32 port = 0x8e4; port <= 0x8f3; ++port) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(std::make_shared<Tegra::Control::AmpereGraphicsState>());
        REQUIRE(engine.CallNativeMethod(system, 0x8e0, 32, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x8e1, 0, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x8e2, 0, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0x8e3, 4, true, {}));
        REQUIRE_FALSE(engine.CallNativeMethod(system, port, 0x12345678, true, {}));
        REQUIRE(engine.GetRegisterValue(0x8e3) == 4);
        REQUIRE(engine.HasNativeMacroFailure());
    }
}

TEST_CASE("Native graphics macro input cannot bypass pending invocation or action guards",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 32, 1ULL << 32, 16, 12};
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    MethodHost host;
    Processor uploader{state};
    Upload(uploader, SumProgram(0xd00), host);
    for (u32 method : {0x52U, 0x53U, 0x54U, 0x55U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(state);
        REQUIRE_FALSE(engine.CallNativeMethod(system, method, 1, true, {}));
        REQUIRE(engine.HasNativeMacroFailure());
    }
    for (u32 target : {0x418800U, 0x419f78U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(state);
        REQUIRE(engine.CallNativeMethod(system, 0xd00, 0, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0xd01, 1, true, {}));
        REQUIRE(engine.CallNativeMethod(system, 0xd02, 1, true, {}));
        REQUIRE_FALSE(engine.CallNativeMethod(system, 0x8c4, target, true, {}));
        REQUIRE(engine.GetRegisterValue(0xd00) == 0);
        REQUIRE_FALSE(engine.GetNativeGraphicsConfig().PolygonOffsetNanIsZero());
        REQUIRE(engine.HasNativeMacroFailure());
    }
    for (const auto [setter, check] :
         std::array<std::pair<u32, u32>, 3>{{{0x58e, 0x5a8}, {0x5a9, 0x5aa}, {0x5e4, 0x5e5}}}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(state);
        REQUIRE(engine.CallNativeMethod(system, setter, 0x40004, true, {}));
        REQUIRE(engine.CallNativeMethod(system, check, 0x40004, true, {}));
        REQUIRE_FALSE(engine.CallNativeMethod(system, check, 0x30003, true, {}));
        REQUIRE(engine.HasNativeMacroFailure());
    }
    {
        Tegra::Engines::Maxwell3D engine{space};
        REQUIRE_FALSE(engine.execution_mask[0xc1]);
        engine.SetNativeMacroState(state);
        REQUIRE_FALSE(engine.CallNativeMethod(system, 0xc1, 1, true, {}));
        REQUIRE(engine.HasNativeMacroFailure());
    }
    {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(state);
        REQUIRE_FALSE(engine.CallNativeMethod(system, 0x44, 0, true, {}));
        REQUIRE(engine.HasNativeMacroFailure());
    }
    for (u32 method : {0xd00U, u32(MAXWELL3D_REG_INDEX(clear_surface)), 0x157U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(state);
        REQUIRE(engine.execution_mask[0xd00]);
        engine.CallMethod(system, 0xe0a, 1, false);
        engine.CallMethod(system, method, 99, true);
        REQUIRE(engine.HasNativeMacroFailure());
        engine.CallMethod(system, 0xe0b, 2, true);
        REQUIRE(engine.GetRegisterValue(0xd00) == 0);
    }
    for (u32 method : {u32(MAXWELL3D_REG_INDEX(clear_surface)), 0x157U}) {
        Tegra::Engines::Maxwell3D engine{space};
        engine.SetNativeMacroState(state);
        const u32 before = engine.GetRegisterValue(method);
        engine.CallMethod(system, method, 99, true);
        REQUIRE(engine.HasNativeMacroFailure());
        REQUIRE(engine.GetRegisterValue(method) == before);
        engine.SetNativeMacroState(nullptr);
        REQUIRE_FALSE(engine.HasNativeMacroFailure());
        REQUIRE_FALSE(engine.execution_mask[0xd00]);
    }
}

TEST_CASE("Native macro calls latch their entry and forward finite instruction budgets",
          "[ipc][video_core]") {
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    Processor first{state}, rebinder{state};
    MethodHost host;
    Upload(first, SumProgram(0xd00), host);
    REQUIRE(first.CallMethod(0xe0a, 4, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(rebinder.CallMethod(0x47, 5, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(rebinder.CallMethod(0x48, 10, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(first.CallMethod(0xe0b, 6, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(host.writes.back() == std::pair<u32, u32>{0xd00, 10});
    Processor incomplete{state};
    REQUIRE(incomplete.CallMethod(0xe0a, 1, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(incomplete.CallMethod(0xe0b, 2, true, host.Bind()) == Processor::Status::Failed);
    REQUIRE(incomplete.GetExecutionResult().error == Engine::Error::IncompleteInstruction);
    REQUIRE(rebinder.CallMethod(0x47, 5, true, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(rebinder.CallMethod(0x48, 7, true, host.Bind()) == Processor::Status::Accepted);
    Processor budget{state};
    REQUIRE(budget.CallMethod(0xe0a, 1, false, host.Bind()) == Processor::Status::Accepted);
    REQUIRE(budget.CallMethod(0xe0b, 2, true, host.Bind(), 2) == Processor::Status::Failed);
    REQUIRE(budget.GetExecutionResult().error == Engine::Error::InstructionLimit);
    REQUIRE(host.writes.back() == std::pair<u32, u32>{0xd00, 3});
}

TEST_CASE("Native compute macros share uploaded programs and retain their own method state",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    auto context = std::make_shared<Tegra::Control::NativeGpuGraphicsContext>(state, space);
    Processor uploader{state};
    MethodHost upload_host;
    Upload(uploader, SumProgram(0xd00), upload_host);
    using Compute = Tegra::Control::NativeCompute;
    Compute first{context}, second{context};
    REQUIRE(first.CallMethod(0xe0a, 8, false, {}));
    REQUIRE(first.HasPendingMacro());
    REQUIRE(second.CallMethod(0xe0a, 3, false, {}));
    REQUIRE(first.CallMethod(0xe0b, 9, true, {}));
    REQUIRE(first.ReadRegister(0xd00) == 17);
    REQUIRE(second.ReadRegister(0xd00) == 0);
    REQUIRE(second.CallMethod(0xe0b, 4, true, {}));
    REQUIRE(second.ReadRegister(0xd00) == 7);
    REQUIRE(first.ReadRegister(0xd00) == 17);
    REQUIRE(first.CallMethod(0xe0a, 1, false, {}));
    REQUIRE_FALSE(first.CallMethod(0xb0, 3, true, {}));
    REQUIRE(first.GetError() == Compute::Error::MacroFailed);
    REQUIRE_FALSE(context->HasQmdFailure());
    REQUIRE_FALSE(first.CallMethod(0xe0b, 1, true, {}));
}

TEST_CASE("Native compute switch configuration retains callbacks without invoking them",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    auto context = std::make_shared<Tegra::Control::NativeGpuGraphicsContext>(
        std::make_shared<Tegra::Control::AmpereGraphicsState>(), space);
    using Compute = Tegra::Control::NativeCompute;
    Compute first{context}, second{context};
    REQUIRE(first.CallMethod(0xd00, 0x12345678, true, {}));
    REQUIRE(first.CallMethod(0x7b, 0x3e3d1, true, {}));
    REQUIRE(first.ReadRegister(0x7b) == 0x3e3d1);
    REQUIRE(first.ReadRegister(0xd00) == 0x12345678);
    REQUIRE(second.ReadRegister(0x7b) == 0);
    REQUIRE(second.CallMethod(0x7b, 0, true, {}));
    REQUIRE(first.ReadRegister(0x7b) == 0x3e3d1);
    for (u32 invalid : {0x3e3d3U, 0x13e3d1U, 0x803d1U, 0x3e801U}) {
        Compute rejected{context};
        REQUIRE(rejected.CallMethod(0x7b, 0x3e3d1, true, {}));
        REQUIRE_FALSE(rejected.CallMethod(0x7b, invalid, true, {}));
        REQUIRE(rejected.ReadRegister(0x7b) == 0x3e3d1);
    }
}

TEST_CASE("Native compute texture cache invalidation executes its backend and propagates failures",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    auto context = std::make_shared<Tegra::Control::NativeGpuGraphicsContext>(
        std::make_shared<Tegra::Control::AmpereGraphicsState>(), space);
    using Compute = Tegra::Control::NativeCompute;
    struct Backend {
        u32 calls{};
        bool success{true}, stopped{}, cancel_in_callback{};
    } backend;
    Compute::Host host{};
    host.qmd.context = &backend;
    host.qmd.should_stop = [](void* opaque) { return static_cast<Backend*>(opaque)->stopped; };
    host.invalidate_caches = [](void* opaque,
                                const Tegra::Control::NativeCacheInvalidation& action) {
        auto& state = *static_cast<Backend*>(opaque);
        ++state.calls;
        REQUIRE(action.texture_data);
        REQUIRE_FALSE((action.instructions || action.global_data || action.constants ||
                       action.samplers || action.texture_headers));
        state.stopped = state.cancel_in_callback;
        return state.success;
    };
    Compute valid{context};
    REQUIRE(valid.CallMethod(0x4a2, 0, true, host));
    REQUIRE(backend.calls == 1);
    for (u32 value : {1U, 0x10U, 0xffffffffU}) {
        Compute rejected{context};
        REQUIRE_FALSE(rejected.CallMethod(0x4a2, value, true, host));
        REQUIRE(backend.calls == 1);
    }
    Compute missing{context};
    REQUIRE_FALSE(missing.CallMethod(0x4a2, 0, true, {}));
    REQUIRE(missing.GetError() == Compute::Error::CacheFailed);
    backend.success = false;
    Compute failed{context};
    REQUIRE_FALSE(failed.CallMethod(0x4a2, 0, true, host));
    REQUIRE(failed.GetError() == Compute::Error::CacheFailed);
    backend.success = true;
    backend.cancel_in_callback = true;
    Compute cancelled{context};
    REQUIRE_FALSE(cancelled.CallMethod(0x4a2, 0, true, host));
    REQUIRE(cancelled.GetError() == Compute::Error::Cancelled);
    REQUIRE(backend.calls == 3);
}

TEST_CASE("Native compute methods fail unsupported actions without invoking legacy compute",
          "[ipc][video_core]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto space = std::make_shared<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    auto state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
    auto context = std::make_shared<Tegra::Control::NativeGpuGraphicsContext>(state, space);
    Tegra::Engines::KeplerCompute compute{*space};
    compute.SetNativeContext(context);
    REQUIRE(compute.execution_mask[0xad]);
    REQUIRE(compute.CallNativeMethod(0xad, 0x10, true));
    REQUIRE_FALSE(compute.CallNativeMethod(0xb0, 3, true));
    REQUIRE(compute.HasNativeMethodFailure());
    REQUIRE(context->HasQmdFailure());
    REQUIRE_FALSE(compute.CallNativeMethod(0xd00, 12, true));
    compute.SetNativeContext(nullptr);
    REQUIRE_FALSE(compute.HasNativeMethodFailure());
    REQUIRE_FALSE(compute.execution_mask[0xad]);
    REQUIRE(compute.execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)]);
    using Compute = Tegra::Control::NativeCompute;
    for (const u32 method : {0xaeU, 0x157U, 0x16aU}) {
        Compute rejected{context};
        REQUIRE_FALSE(rejected.CallMethod(method, 1, true, {}));
        REQUIRE(rejected.GetError() == Compute::Error::UnsupportedMethod);
        REQUIRE(rejected.ReadRegister(method) == 0);
        REQUIRE_FALSE(rejected.CallMethod(0xad, 0x20, true, {}));
    }
    for (const u32 method : {0x6cU, 0x6dU}) {
        Compute missing_setup{context};
        REQUIRE_FALSE(missing_setup.CallMethod(method, 1, true, {}));
        REQUIRE(missing_setup.GetError() == Compute::Error::UploadFailed);
        REQUIRE(missing_setup.ReadRegister(method) == 0);
        REQUIRE_FALSE(missing_setup.CallMethod(0xad, 0x20, true, {}));
    }
    Compute shadow{context};
    REQUIRE(shadow.CallMethod(0x49, 1, true, {}));
    REQUIRE_FALSE(shadow.CallMethod(0x49, 3, true, {}));
    REQUIRE(shadow.ReadRegister(0x49) == 1);
    std::stop_source stop;
    stop.request_stop();
    compute.SetNativeContext(context);
    REQUIRE_FALSE(compute.CallNativeMethod(0xad, 0x10, true, stop.get_token()));
    REQUIRE(compute.HasNativeMethodFailure());

    auto setup = Method();
    Inst read;
    read.alu[0] = {Op::State, Z, I, Z};
    read.immediate[0] = 0x123;
    read.end_next = true;
    Processor uploader{state};
    MethodHost upload_host;
    Upload(uploader, std::array{setup, read, Inst{}}, upload_host);
    Compute unknown_read{context};
    REQUIRE_FALSE(unknown_read.CallMethod(0xe0a, 0, true, {}));
    REQUIRE(unknown_read.GetError() == Compute::Error::MacroFailed);
}
