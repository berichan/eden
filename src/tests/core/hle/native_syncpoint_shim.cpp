// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/service/nvdrv/core/syncpoint_manager.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_shim.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"

TEST_CASE("Native and legacy syncpoint reservations share a global collision-free allocator",
          "[core][ipc]") {
    using Manager = Tegra::Host1x::SyncpointManager;
    Manager manager;
    for (u32 id : {0U, 12U, 26U, 27U, 54U, 55U, 448U, 0xffffffffU}) {
        REQUIRE_FALSE(manager.TryReserveSyncpoint(id));
    }
    REQUIRE(manager.TryReserveSyncpoint(17));
    REQUIRE_FALSE(manager.TryReserveSyncpoint(17));
    auto first = manager.AllocateNativeChannelSyncpoint();
    REQUIRE(first.has_value());
    REQUIRE(first->GetId() == 18);
    manager.ReleaseSyncpoint(17);
    auto second = manager.AllocateNativeChannelSyncpoint();
    REQUIRE(second->GetId() == 17);
    REQUIRE_FALSE(manager.TryReserveSyncpoint(second->GetId()));
    manager.IncrementHost(17);
    manager.IncrementHost(17);
    second.reset();
    REQUIRE(manager.TryReserveSyncpoint(17));
    REQUIRE(manager.GetHostSyncpointValue(17) == 2);
    manager.ReleaseSyncpoint(17);
    first.reset();
    std::vector<Manager::Reservation> leases;
    std::array<bool, Manager::NativeSyncpointCount> seen{};
    while (auto lease = manager.AllocateNativeChannelSyncpoint()) {
        const u32 id = lease->GetId();
        REQUIRE(id >= Manager::NativeChannelSyncpointBase);
        REQUIRE(id < Manager::NativeSyncpointCount);
        REQUIRE_FALSE(Manager::IsFixedSyncpoint(id));
        REQUIRE_FALSE(seen[id]);
        seen[id] = true;
        leases.push_back(std::move(*lease));
    }
    REQUIRE(leases.size() ==
            Manager::NativeSyncpointCount - Manager::NativeChannelSyncpointBase - 4);
    REQUIRE_FALSE(manager.AllocateNativeChannelSyncpoint());
    leases.clear();
    auto reused = manager.AllocateNativeChannelSyncpoint();
    REQUIRE(reused->GetId() == 17);
    REQUIRE(manager.GetHostSyncpointValue(17) == 2);
    auto closing = std::make_unique<Manager>();
    auto final_lease = closing->AllocateNativeChannelSyncpoint();
    REQUIRE(final_lease.has_value());
    closing.reset();
    final_lease.reset();
}

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

TEST_CASE("Legacy driver clients reserve global counters and import their current values",
          "[core][ipc]") {
    using Service::Nvidia::NvCore::SyncpointManager;
    Core::System system;
    system.Initialize();
    auto host = std::make_unique<Tegra::Host1x::Host1x>(system);
    auto& counters = host->GetSyncpointManager();
    counters.IncrementHost(1);
    counters.IncrementHost(1);
    auto native = counters.AllocateNativeChannelSyncpoint();
    REQUIRE(native->GetId() == 17);
    std::vector<u32> allocated;
    {
        SyncpointManager first{*host};
        SyncpointManager second{*host};
        const u32 id = first.AllocateSyncpoint(false);
        REQUIRE(id == 1);
        REQUIRE(first.GetSyncpointFence(id).value == 2);
        REQUIRE_FALSE(counters.TryReserveSyncpoint(id));
        REQUIRE(second.AllocateSyncpoint(false) == 2);
        first.FreeSyncpoint(id);
        REQUIRE(counters.TryReserveSyncpoint(id));
        counters.ReleaseSyncpoint(id);
        for (size_t index = 0; index < 20; ++index) {
            const u32 next = first.AllocateSyncpoint(false);
            REQUIRE(next != 17);
            REQUIRE_FALSE(Tegra::Host1x::SyncpointManager::IsFixedSyncpoint(next));
            allocated.push_back(next);
        }
    }
    for (u32 id : allocated) {
        REQUIRE(counters.TryReserveSyncpoint(id));
        counters.ReleaseSyncpoint(id);
    }
    REQUIRE(counters.TryReserveSyncpoint(2));
    counters.ReleaseSyncpoint(2);
    REQUIRE_FALSE(counters.TryReserveSyncpoint(17));
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
    REQUIRE_FALSE(manager.ReserveSparseIfFree(base, 0x1000));
    REQUIRE_FALSE(manager.ReserveSparseIfFree(base - 0x1000, 0x2000));
    REQUIRE_FALSE(manager.ReserveSparseIfFree(0, 0));
    REQUIRE_FALSE(manager.ReserveSparseIfFree(base + 1, 0x1000));
    REQUIRE_FALSE(manager.ReserveSparseIfFree(1ULL << 32, 0x1000));
    REQUIRE_FALSE(manager.ReserveSparseIfFree((1ULL << 32) - 0x1000, 0x2000));
    REQUIRE_FALSE(manager.ReserveSparseIfFree(~u64{0xfff}, 0x2000));
    REQUIRE_FALSE(manager.MapIntoSparse(base, 0, 0x1000, Tegra::PTEKind::PITCH));
    REQUIRE_FALSE(manager.MapIntoSparse(base + 0x200000, 0, 0x1000, Tegra::PTEKind::PITCH));
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
        REQUIRE_FALSE(manager.WriteTimestampedReleaseChecked(address, 0xffffffff, 0x12345678));
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

TEST_CASE("Checked GPU transfers distinguish sparse holes from faults and read-only registers",
          "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MemoryManager manager{system, backing, 32, 1ULL << 32, 16, 12};
    manager.BindRasterizer(&rasterizer);
    std::array<u8, 32> bytes;
    for (bool big : {false, true}) {
        constexpr GPUVAddr base = 0x8000000;
        REQUIRE(manager.MapSparse(base, 0x10000, big) == base);
        bytes.fill(0xcc);
        REQUIRE(manager.ReadBlockChecked(base + 7, bytes));
        REQUIRE(std::all_of(bytes.begin(), bytes.end(), [](u8 value) { return value == 0; }));
        bytes.fill(0x75);
        REQUIRE(manager.WriteBlockChecked(base + 7, bytes));
        bytes.fill(0xcc);
        REQUIRE_FALSE(manager.ReadBlockChecked(base + 0xfff0, bytes));
        REQUIRE(std::all_of(bytes.begin(), bytes.end(), [](u8 value) { return value == 0xcc; }));
        REQUIRE_FALSE(manager.WriteBlockChecked(base + 0xfff0, bytes));
        manager.Unmap(base, 0x10000);
        REQUIRE_FALSE(manager.ReadBlockChecked(base, bytes));
    }
    REQUIRE(manager.ReadBlockChecked(1ULL << 32, {}));
    REQUIRE(manager.WriteBlockChecked(1ULL << 32, {}));
    REQUIRE_FALSE(manager.ReadBlockChecked((1ULL << 32) + 1, {}));
    REQUIRE_FALSE(manager.ReadBlockChecked((std::numeric_limits<u64>::max)() - 1, bytes));
    REQUIRE_FALSE(manager.WriteBlockChecked((1ULL << 32) - 16, bytes));
    // A GPU PTE alone does not make an absent device backing page accessible.
    REQUIRE(manager.Map(0x8000000, 0x1000, 0x1000, Tegra::PTEKind::PITCH, false) == 0x8000000);
    REQUIRE_FALSE(manager.ReadBlockChecked(0x8000000, bytes));
    manager.Unmap(0x8000000, 0x1000);
    Tegra::Host1x::SyncpointManager syncpoints;
    constexpr GPUVAddr shim = 0x10000000;
    REQUIRE(manager.MapSyncpointShim(shim, syncpoints));
    syncpoints.IncrementHost(0);
    const auto word = std::span{bytes}.first(4);
    REQUIRE(manager.ReadBlockChecked(shim, word));
    u32 value{};
    std::memcpy(&value, bytes.data(), sizeof(value));
    REQUIRE(value == 1);
    syncpoints.IncrementHost(0);
    REQUIRE(manager.ReadBlockChecked(shim, word));
    std::memcpy(&value, bytes.data(), sizeof(value));
    REQUIRE(value == 2);
    REQUIRE_FALSE(manager.WriteBlockChecked(shim, word));
    REQUIRE_FALSE(manager.IsWritableRange(shim, word.size()));
    bytes.fill(0xcc);
    REQUIRE_FALSE(manager.ReadBlockChecked(shim, bytes));
    REQUIRE(bytes.front() == 0xcc);
    REQUIRE_FALSE(manager.ReadBlockChecked(shim + 4, word));
    REQUIRE(manager.MapSparse(shim - 0x1000, 0x1000, false) == shim - 0x1000);
    REQUIRE(manager.ReadBlockChecked(shim - 4, std::span{bytes}.first(8)));
    std::memcpy(&value, bytes.data() + 4, sizeof(value));
    REQUIRE(value == 2);
    REQUIRE(bytes.front() == 0);
    REQUIRE_FALSE(manager.WriteBlockChecked(shim - 4, std::span{bytes}.first(8)));
}
