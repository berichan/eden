// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <limits>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/syncpoint_shim.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"

TEST_CASE("Native syncpoint shim reads completed counters without exposing other registers",
          "[core][ipc]") {
    Tegra::Host1x::SyncpointManager syncpoints;
    Tegra::Host1x::SyncpointShim shim{syncpoints};
    REQUIRE(shim.Count == 448);
    REQUIRE(shim.Stride == 0x10000);
    REQUIRE(shim.Size == 0x1c00000);
    REQUIRE(shim.Read(0, {}));
    for (const u32 id : {0U, 191U, 192U, 447U}) {
        const u64 offset = u64{id} * shim.Stride;
        std::array<u8, 6> output{0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa};
        auto word = std::span{output}.subspan(1, 4);
        syncpoints.IncrementGuest(id);
        REQUIRE(shim.Read(offset, word));
        REQUIRE(output == std::array<u8, 6>{0xaa, 0, 0, 0, 0, 0xaa});
        for (u32 count = 0; count < 257; ++count) {
            syncpoints.IncrementHost(id);
        }
        REQUIRE(shim.Read(offset, word));
        REQUIRE(output == std::array<u8, 6>{0xaa, 1, 1, 0, 0, 0xaa});
        REQUIRE(shim.Read(offset + 1, word.first(2)));
        REQUIRE(output[1] == 1);
        REQUIRE(output[2] == 0);
        output.fill(0xaa);
        for (const u64 bad : {offset + 3, offset + 4, offset + 0xfffc}) {
            REQUIRE_FALSE(shim.Read(bad, word));
            REQUIRE(output == std::array<u8, 6>{0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa});
        }
    }
    std::array<u8, 4> output{0xaa, 0xaa, 0xaa, 0xaa};
    REQUIRE_FALSE(shim.Read(shim.Size, output));
    REQUIRE_FALSE(shim.Read((std::numeric_limits<u64>::max)(), output));
    REQUIRE(output == std::array<u8, 4>{0xaa, 0xaa, 0xaa, 0xaa});
}

TEST_CASE("GPU syncpoint mappings reserve a read-only aperture backed by live counters",
          "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager manager{system, backing, 32, 1ULL << 32, 16, 12};
    Tegra::Host1x::SyncpointManager syncpoints;
    constexpr GPUVAddr base = 0x8000000;
    REQUIRE_FALSE(manager.GetSyncpointShimAddress());
    REQUIRE_FALSE(manager.MapSyncpointShim(base + 1, syncpoints));
    REQUIRE_FALSE(manager.MapSyncpointShim(1ULL << 32, syncpoints));
    REQUIRE_FALSE(manager.MapSyncpointShim((1ULL << 32) - 0x10000, syncpoints));
    REQUIRE(manager.MapSyncpointShim(base, syncpoints));
    REQUIRE(manager.GetSyncpointShimAddress() == base);
    REQUIRE_FALSE(manager.MapSyncpointShim(base, syncpoints));
    REQUIRE_FALSE(manager.GpuToCpuAddress(base));
    REQUIRE(manager.GetPointer(base) == nullptr);
    REQUIRE(manager.Map(base, 0, 0x1000, Tegra::PTEKind::INVALID, false) == 0);
    REQUIRE(manager.MapSparse(base, 0x1000, false) == 0);
    manager.Unmap(base, Tegra::Host1x::SyncpointShim::Size);
    REQUIRE(manager.GetSyncpointShimAddress() == base);
    for (const u32 id : {0U, 191U, 192U, 447U}) {
        const GPUVAddr address = base + u64{id} * Tegra::Host1x::SyncpointShim::Stride;
        REQUIRE(manager.Read<u32>(address) == 0);
        syncpoints.IncrementGuest(id);
        REQUIRE(manager.Read<u32>(address) == 0);
        syncpoints.IncrementHost(id);
        REQUIRE(manager.Read<u32>(address) == 1);
        u32 value{};
        manager.ReadBlockUnsafe(address, &value, sizeof(value));
        REQUIRE(value == 1);
        manager.ReadBlock(address, &value, sizeof(value));
        REQUIRE(value == 1);
        manager.Write<u32>(address, 0xffffffff);
        REQUIRE(manager.Read<u32>(address) == 1);
        const u32 attempted = 0xffffffff;
        manager.WriteBlockUnsafe(address, &attempted, sizeof(attempted));
        REQUIRE(manager.Read<u32>(address) == 1);
    }
}

TEST_CASE("GPU syncpoint reads observe host completion updates across threads", "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager manager{system, backing, 32, 1ULL << 32, 16, 12};
    Tegra::Host1x::SyncpointManager syncpoints;
    constexpr GPUVAddr base = 0x8000000;
    REQUIRE(manager.MapSyncpointShim(base, syncpoints));
    constexpr u32 id = 447;
    std::jthread producer{[&] {
        for (u32 count = 0; count < 1000; ++count) {
            syncpoints.IncrementHost(id);
        }
    }};
    u32 previous{};
    for (u32 count = 0; count < 1000; ++count) {
        const u32 value = manager.Read<u32>(base + u64{id} * 0x10000);
        REQUIRE(value >= previous);
        REQUIRE(value <= 1000);
        previous = value;
    }
    producer.join();
    REQUIRE(manager.Read<u32>(base + u64{id} * 0x10000) == 1000);
}

TEST_CASE("Syncpoint apertures do not replace RAM or sparse reservations", "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager manager{system, backing, 32, 1ULL << 32, 16, 12};
    manager.BindRasterizer(&rasterizer);
    Tegra::Host1x::SyncpointManager syncpoints;
    constexpr GPUVAddr base = 0x8000000;
    constexpr GPUVAddr end = base + Tegra::Host1x::SyncpointShim::Size;
    REQUIRE(manager.Map(base, 0x1000, 0x1000, Tegra::PTEKind::INVALID, false) == base);
    REQUIRE(manager.GpuToCpuAddress(base) == DAddr{0x1000});
    REQUIRE_FALSE(manager.MapSyncpointShim(base, syncpoints));
    REQUIRE(manager.GpuToCpuAddress(base) == DAddr{0x1000});
    manager.Unmap(base, 0x1000);
    REQUIRE_FALSE(manager.GpuToCpuAddress(base));
    REQUIRE(manager.MapSparse(base, 0x1000, false) == base);
    REQUIRE_FALSE(manager.MapSyncpointShim(base, syncpoints));
    manager.Unmap(base, 0x1000);
    REQUIRE(manager.MapSyncpointShim(base, syncpoints));
    for (const auto address : {base - 0x1000, end}) {
        REQUIRE(manager.Map(address, 0x2000, 0x1000, Tegra::PTEKind::INVALID, false) == address);
        REQUIRE(manager.GpuToCpuAddress(address) == DAddr{0x2000});
        manager.Unmap(address, 0x1000);
        REQUIRE_FALSE(manager.GpuToCpuAddress(address));
    }
    REQUIRE(manager.GetSyncpointShimAddress() == base);
    syncpoints.IncrementHost(0);
    REQUIRE(manager.Read<u32>(base) == 1);
}
