// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"

TEST_CASE("Native compute launch validation retains full addresses and rejects truncation",
          "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager space{system, backing, 40, 1ULL << 40, 16, 12};
    Tegra::Engines::KeplerCompute compute{space};
    std::array<u32, 64> descriptor{};
    descriptor[12] = 7;
    descriptor[13] = 2;
    descriptor[14] = 3;
    descriptor[17] = 0x200;
    descriptor[18] = (32U << 16) | 0x30;
    descriptor[19] = (1U << 16) | 2;
    descriptor[20] = 1;
    descriptor[23] = 64;
    descriptor[32] = 0x9000000;
    descriptor[33] = (16U << 19) | 0x81;
    descriptor[48] = 0x8000000;
    descriptor[49] = 0x21;
    descriptor[50] = 32;
    REQUIRE(compute.ConfigureNativeLaunch(descriptor));
    REQUIRE(compute.IsNativeMode());
    REQUIRE(compute.regs.code_loc.Address() == 0x2108000000ULL);
    const auto& launch = compute.launch_description;
    REQUIRE(launch.program_start == 0);
    REQUIRE(launch.grid_dim_x == 7);
    REQUIRE(launch.grid_dim_y == 2);
    REQUIRE(launch.grid_dim_z == 3);
    REQUIRE(launch.block_dim_x == 32);
    REQUIRE(launch.block_dim_y == 2);
    REQUIRE(launch.block_dim_z == 1);
    REQUIRE(launch.shared_alloc == 0x200);
    REQUIRE(launch.local_pos_alloc == 64);
    REQUIRE(launch.local_crs_alloc == 32);
    REQUIRE(launch.const_buffer_config[0].Address() == 0x8109000000ULL);
    REQUIRE(launch.const_buffer_config[0].size == 256);
    const auto saved = launch;
    const std::array invalid_fields{
        std::pair{18U, (32U << 16) | 0x31},
        std::pair{4U, 0x100U},
        std::pair{12U, 0x80000000U},
        std::pair{18U, 0x30U},
        std::pair{19U, 0U},
        std::pair{19U, (32U << 16) | 32},
        std::pair{23U, 0x100000U},
        std::pair{50U, 0x100000U},
        std::pair{49U, 0x100U},
        std::pair{33U, (16U << 19) | 0x100},
    };
    for (const auto [word, value] : invalid_fields) {
        auto invalid = descriptor;
        invalid[word] = value;
        REQUIRE_FALSE(compute.ConfigureNativeLaunch(invalid));
        REQUIRE(std::memcmp(&saved, &compute.launch_description, sizeof(saved)) == 0);
        REQUIRE(compute.regs.code_loc.Address() == 0x2108000000ULL);
    }
    auto invalid = descriptor;
    invalid[32] = 0xfffffff0;
    invalid[33] = (16U << 19) | 0xff;
    REQUIRE_FALSE(compute.ConfigureNativeLaunch(invalid));
    REQUIRE(std::memcmp(&saved, &compute.launch_description, sizeof(saved)) == 0);
    // Disabled descriptor slots are not live addresses.
    auto unused = descriptor;
    unused[35] = 0xffffffff;
    REQUIRE(compute.ConfigureNativeLaunch(unused));
    REQUIRE(compute.launch_description.const_buffer_config[1].Address() == 0);
    REQUIRE_FALSE(compute.DispatchNativeCompute(descriptor));
    auto empty = descriptor;
    empty[12] = 0;
    empty[18] = 0x30;
    empty[19] = 0;
    REQUIRE(compute.DispatchNativeCompute(empty));
    REQUIRE(compute.ConfigureNativeLaunch(descriptor));
    const std::array<u32, 3> max_grid{7, 2, 3};
    const std::array<u32, 3> max_block{1024, 1024, 64};
    REQUIRE(compute.IsNativeLaunchSupported(max_grid, max_block, 1024, 512));
    REQUIRE_FALSE(compute.IsNativeLaunchSupported({6, 2, 3}, max_block, 1024, 512));
    REQUIRE_FALSE(compute.IsNativeLaunchSupported(max_grid, {16, 1024, 64}, 1024, 512));
    REQUIRE_FALSE(compute.IsNativeLaunchSupported(max_grid, max_block, 32, 512));
    REQUIRE_FALSE(compute.IsNativeLaunchSupported(max_grid, max_block, 1024, 256));
    auto too_tall = descriptor;
    too_tall[18] = (1U << 16) | 0x30;
    too_tall[19] = (1024U << 16) | 1;
    REQUIRE(compute.ConfigureNativeLaunch(too_tall));
    REQUIRE_FALSE(compute.IsNativeLaunchSupported(max_grid, max_block, 1024, 512));
}
