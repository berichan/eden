// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "dynarmic/tests/A64/testenv.h"

TEST_CASE("Authenticated returns with instruction authentication disabled", "[core][arm]") {
    for (const u32 instruction : {0xd65f0bffU, 0xd65f0fffU}) {
        for (const u64 target : {0x80ULL, 0x7100000000ULL}) {
            CAPTURE(instruction, target);
            A64TestEnv env;
            env.code_mem = {instruction};
            env.ticks_left = 1;
            Dynarmic::A64::UserConfig config;
            config.callbacks = &env;
            Dynarmic::A64::Jit jit{config};
            std::array<u64, 31> registers{};
            for (size_t i = 0; i < registers.size(); ++i) {
                registers[i] = 0x1000 + i;
                jit.SetRegister(i, registers[i]);
            }
            registers[30] = target;
            jit.SetRegister(30, target);
            jit.SetSP(0x10000);
            jit.SetPstate(0xa0000000);
            jit.SetPC(0);

            jit.Step();

            REQUIRE(jit.GetPC() == target);
            REQUIRE(jit.GetSP() == 0x10000);
            REQUIRE(jit.GetPstate() == 0xa0000000);
            for (size_t i = 0; i < registers.size(); ++i) {
                REQUIRE(jit.GetRegister(i) == registers[i]);
            }
        }
    }
}

TEST_CASE("Authenticated returns complete a linked subroutine call", "[core][arm]") {
    for (const bool use_key_b : {false, true}) {
        for (const u64 base : {0ULL, 0x7100000000ULL}) {
            for (const auto optimizations : {Dynarmic::no_optimizations,
                                            Dynarmic::all_safe_optimizations}) {
                CAPTURE(use_key_b, base, static_cast<u32>(optimizations));
                A64TestEnv env;
                env.code_mem_start_address = base;
                env.code_mem = {
                    0x94000004, // BL subroutine
                    0x91000400, // ADD X0, X0, #1
                    0x14000000, // B .
                    0xd503201f, // NOP
                    use_key_b ? 0xd503237fU : 0xd503233fU, // PACIBSP / PACIASP
                    0x91000800, // ADD X0, X0, #2
                    use_key_b ? 0xd65f0fffU : 0xd65f0bffU, // RETAB / RETAA
                };
                env.ticks_left = 6;
                Dynarmic::A64::UserConfig config;
                config.callbacks = &env;
                config.optimizations = optimizations;
                Dynarmic::A64::Jit jit{config};
                jit.SetPC(base);
                jit.SetSP(0x10000);

                jit.Run();

                REQUIRE(jit.GetPC() == base + 8);
                REQUIRE(jit.GetRegister(0) == 3);
                REQUIRE(jit.GetRegister(30) == base + 4);
                REQUIRE(jit.GetSP() == 0x10000);
                REQUIRE_FALSE(env.code_mem_modified_by_guest);
            }
        }
    }
}
