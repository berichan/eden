// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include "common/common_types.h"
#include "common/swap.h"
#include "core/hle/service/nvdrv/nvdata.h"

namespace Service::Nvidia::Devices {

// Native ctrl-gpu replaces the legacy counts at 0x70 with signed ioctl interface levels.
// Hardware properties do not imply that every advertised operation is implemented by HLE.
// Protocol reference: https://switchbrew.org/wiki/NV_services#NVGPU_GPU_IOCTL_GET_CHARACTERISTICS
struct NativeGpuCharacteristics {
    u32_le arch{0x170};
    u32_le impl{0xe};
    u32_le rev{0xa1};
    u32_le num_gpc{1};
    u64_le l2_cache_size{0x100000};
    u64_le on_board_video_memory_size{};
    u32_le num_tpc_per_gpc{6};
    u32_le bus_type{0x20};
    u32_le big_page_size{};
    u32_le compression_page_size{0x10000};
    u32_le pde_coverage_bit_count{0x15};
    u32_le available_big_page_sizes{};
    u32_le gpc_mask{1};
    u32_le sm_arch_sm_version{0x808};
    u32_le sm_arch_spa_version{0x806};
    u32_le sm_arch_warp_count{0x60};
    u32_le gpu_va_bit_count{0x28};
    u32_le reserved{};
    u64_le flags{0x935faf1edc0155};
    u32_le twod_class{0x902d};
    u32_le threed_class{0xc797};
    u32_le compute_class{0xc7c0};
    u32_le gpfifo_class{0xc76f};
    u32_le inline_to_memory_class{0xa140};
    u32_le dma_copy_class{0xc7b5};
    s16_le gpu_ioctl_nr_last{0x1f};
    s16_le tsg_ioctl_nr_last{0xf};
    s16_le dbg_gpu_ioctl_nr_last{0x2b};
    s16_le channel_ioctl_nr_last{0x21};
    s16_le as_ioctl_nr_last{0xf};
    std::array<s16_le, 3> unsupported_interfaces{-1, -1, -1};
    u32_le max_fbps_count{};
    u32_le fbp_en_mask{1};
    u32_le emc_en_mask{1};
    u32_le max_ltc_per_fbp{1};
    u32_le max_lts_per_ltc{4};
    u32_le max_tex_per_tpc{};
    u32_le max_gpc_count{1};
    std::array<u32_le, 2> deprecated_rop_l2_en_masks{};
    std::array<char, 12> chipname{'g', 'a', '1', '0', 'f'};
    u32_le unknown_b0{2};
    u32_le unknown_b4{0x40};
    u32_le unknown_b8{3};
    u32_le unknown_bc{7};
    u32_le unknown_c0{1};
    u32_le unknown_c4{1};
    u64_le memory_limit{};
    u64_le reserved_d0{};
};
static_assert(sizeof(NativeGpuCharacteristics) == 0xd8);
static_assert(offsetof(NativeGpuCharacteristics, gpu_ioctl_nr_last) == 0x70);
static_assert(offsetof(NativeGpuCharacteristics, as_ioctl_nr_last) == 0x78);
static_assert(offsetof(NativeGpuCharacteristics, chipname) == 0xa4);
static_assert(offsetof(NativeGpuCharacteristics, unknown_b0) == 0xb0);
static_assert(offsetof(NativeGpuCharacteristics, memory_limit) == 0xc8);

inline NvResult ReadNativeGpuCharacteristics(std::span<const u8> input, std::span<u8> output,
                                             std::span<u8> inline_output) {
    constexpr size_t RequestSize = 0xe0;
    if (input.size() < RequestSize || output.size() < RequestSize) {
        return NvResult::InvalidSize;
    }
    u64_le requested_size{};
    std::memcpy(&requested_size, input.data(), sizeof(requested_size));
    std::memmove(output.data(), input.data(), RequestSize);
    const NativeGpuCharacteristics characteristics{};
    const u64_le full_size{sizeof(characteristics)};
    std::memcpy(output.data(), &full_size, sizeof(full_size));
    const auto copy_size = static_cast<size_t>(
        std::min<u64>({requested_size, inline_output.size(), sizeof(characteristics)}));
    if (copy_size != 0) {
        std::memcpy(inline_output.data(), &characteristics, copy_size);
    }
    return NvResult::Success;
}

struct NativeGpuEngineInfo {
    u32_le engine_type;
    u32_le instance;
    s32_le runlist_id;
    u32_le reserved;
};
static_assert(sizeof(NativeGpuEngineInfo) == 0x10);

inline NvResult ReadNativeGpuEngineInfo(std::span<const u8> input, std::span<u8> output,
                                        std::span<u8> inline_output) {
    constexpr size_t RequestSize = 0x10;
    if (input.size() < RequestSize || output.size() < RequestSize) {
        return NvResult::InvalidSize;
    }
    // Logical graphics/compute and copy runlists. Submission still goes through the device
    // handlers, which must reject unsupported operations instead of completing work here.
    // Type 0 is graphics; type 2 is copy on a separate runlist. Type 1 would describe
    // a copy engine sharing the graphics runlist and cannot identify the graphics engine.
    const std::array<NativeGpuEngineInfo, 2> engines{{{0, 0, 0, 0}, {2, 0, 1, 0}}};
    std::memmove(output.data(), input.data(), RequestSize);
    const u32_le full_size{sizeof(engines)};
    std::memcpy(output.data(), &full_size, sizeof(full_size));
    const auto count = std::min(inline_output.size() / sizeof(NativeGpuEngineInfo), engines.size());
    if (count != 0) {
        std::memcpy(inline_output.data(), engines.data(), count * sizeof(NativeGpuEngineInfo));
    }
    return NvResult::Success;
}

} // namespace Service::Nvidia::Devices
