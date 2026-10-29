// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/atomic_ops.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/tests/A64/testenv.h"

namespace {

class AtomicTestEnv final : public A64TestEnv {
public:
    explicit AtomicTestEnv(u64& memory_) : memory{memory_} {}
    u64& memory;
    u64 last_address{};
    size_t last_size{};
    u32 operations{};
    Dynarmic::A64::Jit* abort_jit{};

    template <typename T>
    T CompareExchange(u64 address, T expected, T value) {
        last_address = address;
        last_size = sizeof(T);
        ++operations;
        if (abort_jit) {
            abort_jit->HaltExecution(Dynarmic::HaltReason::MemoryAbort);
            return 0;
        }
        T observed{};
        static_cast<void>(
            Common::AtomicCompareAndSwap(reinterpret_cast<T*>(&memory), value, expected, observed));
        return observed;
    }

    u8 MemoryCompareExchange8(u64 a, u8 e, u8 v) override {
        return CompareExchange(a, e, v);
    }
    u16 MemoryCompareExchange16(u64 a, u16 e, u16 v) override {
        return CompareExchange(a, e, v);
    }
    u32 MemoryCompareExchange32(u64 a, u32 e, u32 v) override {
        return CompareExchange(a, e, v);
    }
    u64 MemoryCompareExchange64(u64 a, u64 e, u64 v) override {
        return CompareExchange(a, e, v);
    }

    u64 MemoryRead(u64, size_t size) override {
        u64 observed{};
        static_cast<void>(Common::AtomicCompareAndSwap(&memory, u64{}, u64{}, observed));
        return size == 8 ? observed : observed & ((1ULL << (size * 8)) - 1);
    }

    bool MemoryWriteExclusive64(u64, u64 value, u64 expected) override {
        return Common::AtomicCompareAndSwap(&memory, value, expected);
    }
};

constexpr u32 EncodeCas(u32 size, u32 ordering, u32 rs, u32 rn, u32 rt) {
    return 0x08a07c00U | (size << 30) | ((ordering & 1) << 22) | ((ordering >> 1) << 15) |
           (rs << 16) | (rn << 5) | rt;
}

} // namespace

TEST_CASE("Compare-and-exchange preserves register and memory semantics", "[core][arm]") {
    constexpr std::array<std::array<u32, 3>, 8> registers{{
        {1, 0, 2},
        {1, 31, 2},
        {0, 0, 2},
        {1, 0, 1},
        {1, 0, 0},
        {31, 0, 2},
        {1, 0, 31},
        {31, 31, 31},
    }};
    for (const u32 size : {0U, 1U, 2U, 3U}) {
        const u64 mask = size == 3 ? ~0ULL : (1ULL << (8 << size)) - 1;
        for (const u32 ordering : {0U, 1U, 2U, 3U}) {
            for (const auto regs : registers) {
                for (const bool match : {false, true}) {
                    const auto [rs, rn, rt] = regs;
                    CAPTURE(size, ordering, rs, rn, rt, match);
                    u64 memory{};
                    AtomicTestEnv env{memory};
                    env.code_mem = {EncodeCas(size, ordering, rs, rn, rt)};
                    env.ticks_left = 1;
                    Dynarmic::A64::UserConfig config;
                    config.callbacks = &env;
                    Dynarmic::A64::Jit jit{config};
                    jit.SetPC(0);
                    jit.SetSP(0x2000);
                    jit.SetPstate(0xa0000000);
                    jit.SetRegister(0, 0x1000);
                    jit.SetRegister(1, 0xfedcba9876543281);
                    jit.SetRegister(2, 0x123456789abcdef0);
                    const auto before = jit.GetRegisters();
                    const u64 address = rn == 31 ? jit.GetSP() : before[rn];
                    const u64 expected = rs == 31 ? 0 : before[rs] & mask;
                    const u64 value = rt == 31 ? 0 : before[rt] & mask;
                    const u64 observed = match ? expected : expected ^ 1;
                    memory = (0x5555555555555555ULL & ~mask) | observed;
                    const u64 untouched = memory & ~mask;

                    jit.Step();

                    REQUIRE(env.operations == 1);
                    REQUIRE(env.last_address == address);
                    REQUIRE(env.last_size == (1ULL << size));
                    REQUIRE(memory == (untouched | (match ? value : observed)));
                    REQUIRE(jit.GetPC() == 4);
                    REQUIRE(jit.GetSP() == 0x2000);
                    REQUIRE(jit.GetPstate() == 0xa0000000);
                    for (size_t i = 0; i < before.size(); ++i) {
                        REQUIRE(jit.GetRegister(i) == (i == rs ? observed : before[i]));
                    }
                }
            }
        }
    }
}

TEST_CASE("Compare-and-exchange abort leaves the compared register intact", "[core][arm]") {
    u64 memory = 7;
    AtomicTestEnv env{memory};
    env.code_mem = {EncodeCas(3, 3, 1, 0, 2)};
    env.ticks_left = 1;
    Dynarmic::A64::UserConfig config;
    config.callbacks = &env;
    config.check_halt_on_memory_access = false;
    Dynarmic::A64::Jit jit{config};
    env.abort_jit = &jit;
    jit.SetPC(0);
    jit.SetRegister(0, 0x1000);
    jit.SetRegister(1, 7);
    jit.SetRegister(2, 8);

    const auto halt = jit.Step();

    REQUIRE(static_cast<u32>(halt & Dynarmic::HaltReason::MemoryAbort) != 0);
    REQUIRE(jit.GetRegister(1) == 7);
    REQUIRE(jit.GetPC() == 0);
    REQUIRE(memory == 7);
}

TEST_CASE("Concurrent guest compare-and-exchange does not lose updates", "[core][arm]") {
    u64 memory{};
    auto increment = [&] {
        AtomicTestEnv env{memory};
        env.code_mem = {EncodeCas(3, 3, 0, 1, 2)};
        Dynarmic::A64::UserConfig config;
        config.callbacks = &env;
        Dynarmic::A64::Jit jit{config};
        jit.SetRegister(1, 0x1000);
        u64 expected{};
        for (u32 i = 0; i < 500; ++i) {
            for (;;) {
                jit.SetPC(0);
                jit.SetRegister(0, expected);
                jit.SetRegister(2, expected + 1);
                env.ticks_left = 1;
                jit.Step();
                const u64 observed = jit.GetRegister(0);
                if (observed == expected) {
                    ++expected;
                    break;
                }
                expected = observed;
            }
        }
    };
    std::thread first{increment};
    std::thread second{increment};
    first.join();
    second.join();
    REQUIRE(memory == 1000);
}

TEST_CASE("Atomic exchanges invalidate a stale exclusive reservation", "[core][arm]") {
    u64 memory = 37;
    Dynarmic::ExclusiveMonitor monitor{2};
    AtomicTestEnv exclusive{memory};
    AtomicTestEnv atomic{memory};
    exclusive.code_mem = {0xc85f7c20, 0xc8037c22}; // LDXR X0,[X1]; STXR W3,X2,[X1]
    atomic.code_mem = {EncodeCas(3, 0, 0, 1, 2)};
    Dynarmic::A64::UserConfig config;
    config.callbacks = &exclusive;
    config.global_monitor = &monitor;
    config.processor_id = 0;
    Dynarmic::A64::Jit load_store{config};
    config.callbacks = &atomic;
    config.processor_id = 1;
    Dynarmic::A64::Jit compare{config};
    load_store.SetPC(0);
    load_store.SetRegister(1, 0x1000);
    load_store.SetRegister(2, 99);
    exclusive.ticks_left = 2;
    load_store.Step();
    REQUIRE(load_store.GetRegister(0) == 37);
    for (const auto values : {std::array<u64, 2>{37, 58}, std::array<u64, 2>{58, 37}}) {
        compare.SetPC(0);
        compare.SetRegister(0, values[0]);
        compare.SetRegister(1, 0x1000);
        compare.SetRegister(2, values[1]);
        atomic.ticks_left = 1;
        compare.Step();
        REQUIRE(compare.GetRegister(0) == values[0]);
    }
    load_store.Step();
    REQUIRE(load_store.GetRegister(3) == 1);
    REQUIRE(memory == 37);
}

TEST_CASE("LSE arithmetic and bit operations are atomic at every width", "[core][arm]") {
    for (const u32 size : {0U, 1U, 2U, 3U}) {
        const u32 bits = 8U << size;
        const u64 mask = size == 3 ? ~0ULL : (1ULL << bits) - 1;
        const u64 sign = 1ULL << (bits - 1);
        const u64 positive_max = sign - 1;
        const std::array<u64, 9> negative_one_results{0,    mask - 1, mask - 1, mask, 1,
                                                      mask, mask,     1,        1};
        const std::array<u64, 9> boundary_results{
            mask, sign, mask, mask, positive_max, sign, sign, positive_max, positive_max};
        for (const u32 ordering : {0U, 1U, 2U, 3U}) {
            for (u32 operation = 0; operation <= 8; ++operation) {
                CAPTURE(size, ordering, operation);
                u64 memory{};
                AtomicTestEnv env{memory};
                env.code_mem = {0x38200020U | (size << 30) | ((ordering & 1) << 23) |
                                ((ordering >> 1) << 22) | (2U << 16) | (operation << 12)};
                Dynarmic::A64::UserConfig config;
                config.callbacks = &env;
                Dynarmic::A64::Jit jit{config};
                for (const bool boundary : {false, true}) {
                    const u64 old = boundary ? sign : mask;
                    memory = (0x5555555555555555ULL & ~mask) | old;
                    const u64 untouched = memory & ~mask;
                    env.ticks_left = 1;
                    jit.SetPC(0);
                    jit.SetRegister(0, ~0ULL);
                    jit.SetRegister(1, 0x1000);
                    jit.SetRegister(2, boundary ? positive_max : 1);
                    jit.SetPstate(0xa0000000);

                    jit.Step();

                    REQUIRE(jit.GetRegister(0) == old);
                    REQUIRE(memory == (untouched | (boundary ? boundary_results[operation]
                                                             : negative_one_results[operation])));
                    REQUIRE(env.last_size == (1ULL << size));
                    REQUIRE(env.last_address == 0x1000);
                    REQUIRE(jit.GetRegister(1) == 0x1000);
                    REQUIRE(jit.GetPstate() == 0xa0000000);
                }
            }
        }
    }
}

TEST_CASE("Concurrent atomic adds retain updates when the result is discarded", "[core][arm]") {
    u64 memory{};
    auto increment = [&] {
        AtomicTestEnv env{memory};
        env.code_mem = {0xf8e0003f}; // LDADDAL X0,XZR,[X1]
        Dynarmic::A64::UserConfig config;
        config.callbacks = &env;
        Dynarmic::A64::Jit jit{config};
        jit.SetRegister(0, 1);
        jit.SetRegister(1, 0x1000);
        for (u32 i = 0; i < 500; ++i) {
            jit.SetPC(0);
            env.ticks_left = 1;
            jit.Step();
        }
    };
    std::thread first{increment};
    std::thread second{increment};
    first.join();
    second.join();
    REQUIRE(memory == 1000);
}

TEST_CASE("Exclusive monitor keeps six independent core reservations", "[core][arm][topology]") {
    Dynarmic::ExclusiveMonitor monitor{6};
    REQUIRE(monitor.GetProcessorCount() == 6);
    for (size_t core = 0; core < 6; ++core) {
        REQUIRE(monitor.ReadAndMark<u64>(core, 0x1000 + core * 8, [core] { return core + 100; }) ==
                core + 100);
    }
    for (size_t core = 0; core < 6; ++core) {
        REQUIRE(monitor.DoExclusiveOperation<u64>(core, 0x1000 + core * 8,
                                                  [core](u64 old) { return old == core + 100; }));
    }
    monitor.ReadAndMark<u64>(0, 0x2000, [] { return 37; });
    monitor.ReadAndMark<u64>(5, 0x2000, [] { return 37; });
    REQUIRE(monitor.DoExclusiveOperation<u64>(5, 0x2000, [](u64 old) { return old == 37; }));
    REQUIRE_FALSE(monitor.DoExclusiveOperation<u64>(0, 0x2000, [](u64) { return true; }));
}

TEST_CASE("Every guest CPU can execute an exclusive load and store", "[core][arm][topology]") {
    u64 memory = 37;
    Dynarmic::ExclusiveMonitor monitor{6};
    for (size_t core = 0; core < 6; ++core) {
        AtomicTestEnv env{memory};
        env.code_mem = {0xc85f7c20, 0xc8037c22}; // LDXR X0,[X1]; STXR W3,X2,[X1]
        Dynarmic::A64::UserConfig config;
        config.callbacks = &env;
        config.global_monitor = &monitor;
        config.processor_id = static_cast<u8>(core);
        Dynarmic::A64::Jit jit{config};
        jit.SetRegister(1, 0x1000);
        jit.SetRegister(2, 100 + core);
        env.ticks_left = 2;
        jit.Step();
        REQUIRE(jit.GetRegister(0) == memory);
        jit.Step();
        REQUIRE(jit.GetRegister(3) == 0);
        REQUIRE(memory == 100 + core);
    }
}
