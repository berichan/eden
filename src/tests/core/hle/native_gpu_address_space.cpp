// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/service/gsv/gpu_address_space_pool.h"
#include "core/hle/service/nvdrv/devices/native_gpu_va_range.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/syncpoint_shim.h"
#include "video_core/memory_manager.h"

TEST_CASE("Native GPU address space handles retain independent memory managers", "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto pool = std::make_shared<Service::GSV::GpuAddressSpacePool>();
    const std::weak_ptr weak_pool{pool};
    std::array<size_t, 8> identifiers{};
    u32 handle{0xffffffff};
    REQUIRE(pool->Add(&handle, nullptr, 0) == Service::GSV::GpuAddressSpacePool::InvalidArgument);
    REQUIRE(handle == 0xffffffff);
    REQUIRE(pool->Get(0x02000000) == nullptr);
    REQUIRE_FALSE(pool->GetFlags(0x02000000));
    for (u32 index = 0; index < 8; ++index) {
        REQUIRE(pool->HasCapacity());
        // Smaller real page tables keep this ownership/capacity test independent of guest RAM.
        auto manager =
            std::make_unique<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
        identifiers[index] = manager->GetID();
        REQUIRE(pool->Add(&handle, std::move(manager), -static_cast<s32>(index)).IsSuccess());
        REQUIRE(handle == (0x02000000U | index));
        REQUIRE(pool->Get(handle) != nullptr);
        REQUIRE(pool->Get(handle)->GetID() == identifiers[index]);
        REQUIRE(pool->GetFlags(handle) == -static_cast<s32>(index));
        REQUIRE_FALSE(pool->Get(handle)->GpuToCpuAddress(0x8000000));
        u64 first{};
        u64 last{};
        REQUIRE(pool->QueryPartitionRange(&first, &last, handle,
                                          Service::GSV::GpuAddressSpacePool::Partition::User)
                    .IsSuccess());
        REQUIRE(first == 0x8000000ULL);
        REQUIRE(last == 0xbfffffffffULL);
        REQUIRE(pool->QueryPartitionRange(&first, &last, handle,
                                          Service::GSV::GpuAddressSpacePool::Partition::Driver)
                    .IsSuccess());
        REQUIRE(first == 0xc000000000ULL);
        REQUIRE(last == 0xc0ffffffffULL);
        REQUIRE(pool->QueryPartitionRange(&first, &last, handle,
                                          Service::GSV::GpuAddressSpacePool::Partition::Gsv)
                    .IsSuccess());
        REQUIRE(first == (index == 0 ? 0xc800000000ULL : 0xc900000000ULL));
        REQUIRE(last == (index == 0 ? 0xc8ffffffffULL : 0xcfffffffffULL));
        for (u32 previous = 0; previous < index; ++previous) {
            REQUIRE(identifiers[previous] != identifiers[index]);
        }
    }
    REQUIRE_FALSE(pool->HasCapacity());
    handle = 0xffffffff;
    REQUIRE(
        pool->Add(&handle,
                  std::make_unique<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12),
                  0) == Service::GSV::GpuAddressSpacePool::OutOfAddressSpaces);
    REQUIRE(handle == 0xffffffff);
    for (const u32 invalid : {0U, 0x01000000U, 0x02000008U, 0x02000100U, 0xffffffffU}) {
        REQUIRE(pool->Get(invalid) == nullptr);
        REQUIRE_FALSE(pool->GetFlags(invalid));
        u64 first{1};
        u64 last{2};
        REQUIRE(pool->QueryPartitionRange(&first, &last, invalid,
                                          Service::GSV::GpuAddressSpacePool::Partition::User) ==
                Service::GSV::GpuAddressSpacePool::InvalidAddressSpace);
        REQUIRE(first == 1);
        REQUIRE(last == 2);
    }
    auto worker_reference = pool;
    pool.reset();
    REQUIRE_FALSE(weak_pool.expired());
    REQUIRE(worker_reference->Get(0x02000007)->GetID() == identifiers[7]);
    worker_reference.reset();
    REQUIRE(weak_pool.expired());
}

TEST_CASE("Native driver bindings keep the canonical address space alive after context release",
          "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    auto pool = std::make_shared<Service::GSV::GpuAddressSpacePool>();
    auto manager = std::make_unique<Tegra::MemoryManager>(system, backing, 32, 1ULL << 32, 16, 12);
    const auto id = manager->GetID();
    u32 handle{};
    REQUIRE(pool->Add(&handle, std::move(manager), 0).IsSuccess());
    auto first_binding = pool->Acquire(handle);
    auto second_binding = pool->Acquire(handle);
    REQUIRE(first_binding != nullptr);
    REQUIRE(first_binding.get() == pool->Get(handle));
    REQUIRE(second_binding.get() == first_binding.get());
    Service::GSV::GpuAddressSpacePool foreign;
    REQUIRE(foreign.Acquire(handle) == nullptr);
    REQUIRE(pool->Acquire(handle + 1) == nullptr);
    REQUIRE(pool->Acquire(0xffffffff) == nullptr);
    const std::weak_ptr weak_manager{first_binding};
    pool.reset();
    REQUIRE(first_binding->GetID() == id);
    REQUIRE(second_binding->GetID() == id);
    REQUIRE_FALSE(first_binding->GpuToCpuAddress(0x8000000));
    first_binding.reset();
    REQUIRE_FALSE(weak_manager.expired());
    second_binding.reset();
    REQUIRE(weak_manager.expired());
}

TEST_CASE("Native GPU page tables cover the full 40-bit address range without initial mappings",
          "[core][ipc]") {
    Core::System system;
    Core::DeviceMemory memory;
    Tegra::MaxwellDeviceMemoryManager backing{memory};
    Tegra::MemoryManager manager{system, backing, 40, 1ULL << 40, 16, 12};
    for (const u64 address :
         {0ULL, 0x8000000ULL, 0xbfffffffffULL, 0xc000000000ULL, 0xc8ffffffffULL, 0xffffffffffULL}) {
        CAPTURE(address);
        REQUIRE(manager.IsWithinGPUAddressRange(address));
        REQUIRE_FALSE(manager.GpuToCpuAddress(address));
    }
    REQUIRE_FALSE(manager.IsWithinGPUAddressRange(1ULL << 40));
    REQUIRE_FALSE(manager.GpuToCpuAddress(1ULL << 40));
    Tegra::Host1x::SyncpointManager syncpoints;
    constexpr GPUVAddr high = 0xc000000000;
    REQUIRE(manager.MapSyncpointShim(high, syncpoints));
    syncpoints.IncrementHost(447);
    REQUIRE(manager.Read<u32>(high + u64{447} * 0x10000) == 1);
    REQUIRE(manager.GetPointer(high) == nullptr);
    REQUIRE(manager.GetSyncpointShimAddress() == high);
}

TEST_CASE("Native driver VA ranges preserve full-width partition limits and reject invalid bounds",
          "[core][ipc]") {
    using Service::Nvidia::Devices::NativeGpuVaRange;
    constexpr u64 first = 0x8000000;
    constexpr u64 last = 0xbfffffffff;
    const auto full = NativeGpuVaRange::Create(first, last, 0, 0);
    REQUIRE(full.has_value());
    REQUIRE(full->first == first);
    REQUIRE(full->end == 0xc000000000);
    REQUIRE((full->end - full->first) / 0x1000 == 0xbff8000ULL);
    const auto partial = NativeGpuVaRange::Create(first, last, first + 0x1000, last - 0xfff);
    REQUIRE(partial.has_value());
    REQUIRE(partial->first == first + 0x1000);
    REQUIRE(partial->end == last - 0xfff);
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, last, first - 0x1000, 0));
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, last, 0, last + 0x1001));
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, last, first + 1, 0));
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, last, 0, last));
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, last, first + 0x1000, first));
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, last, first, first));
    REQUIRE_FALSE(NativeGpuVaRange::Create(last, first, 0, 0));
    REQUIRE_FALSE(NativeGpuVaRange::Create(first, ~u64{0}, 0, 0));
}
