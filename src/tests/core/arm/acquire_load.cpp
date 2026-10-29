// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/tests/A64/testenv.h"

namespace {

class AcquireLoadTestEnv final : public A64TestEnv {
public:
    static constexpr u64 Value = 0xfedcba9876543281;
    u64 read_address{};
    size_t read_size{};
    u32 read_count{};

    u64 MemoryRead(u64 address, size_t size) override {
        read_address = address;
        read_size = size;
        ++read_count;
        return size == 8 ? Value : Value & ((1ULL << (size * 8)) - 1);
    }
};

} // namespace

TEST_CASE("RCpc acquire loads preserve width and register semantics", "[core][arm]") {
    for (const u32 size : {0U, 1U, 2U, 3U}) {
        for (const bool use_sp : {false, true}) {
            for (const bool discard_result : {false, true}) {
                CAPTURE(size, use_sp, discard_result);
                const u32 rn = use_sp ? 31 : 1;
                const u32 rt = discard_result ? 31 : 2;
                AcquireLoadTestEnv env;
                env.code_mem = {0x38bfc000U | (size << 30) | (rn << 5) | rt};
                env.ticks_left = 1;
                Dynarmic::A64::UserConfig config;
                config.callbacks = &env;
                Dynarmic::A64::Jit jit{config};
                jit.SetPC(0);
                jit.SetSP(0x2000);
                jit.SetRegister(1, 0x1000);
                jit.SetRegister(2, 0xffffffffffffffff);
                jit.SetPstate(0xa0000000);

                jit.Step();

                REQUIRE(env.read_count == 1);
                REQUIRE(env.read_address == (use_sp ? 0x2000ULL : 0x1000ULL));
                REQUIRE(env.read_size == (1ULL << size));
                REQUIRE(jit.GetPC() == 4);
                REQUIRE(jit.GetRegister(1) == 0x1000);
                REQUIRE(jit.GetSP() == 0x2000);
                REQUIRE(jit.GetPstate() == 0xa0000000);
                const u64 expected = size == 3
                                         ? AcquireLoadTestEnv::Value
                                         : AcquireLoadTestEnv::Value & ((1ULL << (8 << size)) - 1);
                REQUIRE(jit.GetRegister(2) == (discard_result ? 0xffffffffffffffff : expected));
            }
        }
    }
}
