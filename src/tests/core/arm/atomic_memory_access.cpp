// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <catch2/catch_test_macros.hpp>

#include "common/host_memory.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_system_resource.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"

TEST_CASE("Atomic access permissions follow guest mappings without guest scheduling",
          "[core][arm]") {
    const bool multicore = Settings::values.use_multi_core.GetValue();
    const auto fastmem = Settings::values.cpuopt_fastmem.GetValue();
    const auto accuracy = Settings::values.cpu_accuracy.GetValue();
    Settings::values.use_multi_core = false;
    Settings::values.cpu_accuracy = Settings::CpuAccuracy::Debugging;
    Settings::values.cpuopt_fastmem = false;
    SCOPE_EXIT {
        Settings::values.use_multi_core = multicore;
        Settings::values.cpuopt_fastmem = fastmem;
        Settings::values.cpu_accuracy = accuracy;
    };
    Core::System system;
    system.Initialize();
    auto& kernel = system.Kernel();
    kernel.Initialize();
    SCOPE_EXIT {
        kernel.Shutdown();
    };
    Kernel::KProcess process{kernel};
    Kernel::KThread thread{kernel};
    Kernel::KAutoObject::Create(&process);
    kernel.SetCurrentEmuThread(&thread);
    REQUIRE(Kernel::KThread::InitializeDummyThread(system, &thread, &process).IsSuccess());
    SCOPE_EXIT {
        kernel.SetCurrentEmuThread(&thread);
        process.UnregisterThread(kernel, &thread);
        process.Close(kernel);
        kernel.UnregisterKernelObject(&thread);
        kernel.UnregisterKernelObject(&process);
        kernel.SetCurrentEmuThread(nullptr);
    };
    auto& table = process.GetPageTable();
    auto& memory = process.GetMemory();
    REQUIRE(table
                .Initialize(Kernel::Svc::CreateProcessFlag::AddressSpace32Bit, false, false, false,
                            Kernel::KMemoryManager::Pool::Application, 0x200000, Kernel::PageSize,
                            &kernel.GetAppSystemResource(), kernel.GetSystemResourceLimit(), memory,
                            0)
                .IsSuccess());
    memory.SetCurrentPageTable(process);
    SCOPE_EXIT {
        kernel.SetCurrentEmuThread(&thread);
        table.Finalize();
    };
    REQUIRE(table.GetImpl().fastmem_arena == nullptr);
    REQUIRE(table.SetMaxHeapSize(4 * Kernel::PageSize).IsSuccess());
    Kernel::KProcessAddress address;
    REQUIRE(table.SetHeapSize(&address, 4 * Kernel::PageSize).IsSuccess());
    const u64 base = GetInteger(address);
    const u64 end = base + 4 * Kernel::PageSize;
    Core::DynarmicExclusiveMonitor monitor{memory, 1};
    Core::ArmDynarmic64 cpu{system, true, &process, monitor, 0};
    Core::DynarmicCallbacks64 callbacks{cpu, &process};

    SECTION("callbacks preserve CAS results and reject inaccessible or unaligned memory") {
        // JIT callbacks must work with no schedulable guest thread or guest lock acquisition.
        kernel.SetCurrentEmuThread(nullptr);
        const auto exercise = [&]<typename T>(u64 location, T initial, T replacement) {
            memory.Write64(location, initial);
            const auto cas = [&](T expected, T value) -> T {
                if constexpr (sizeof(T) == 1) {
                    return callbacks.MemoryCompareExchange8(location, expected, value);
                } else if constexpr (sizeof(T) == 2) {
                    return callbacks.MemoryCompareExchange16(location, expected, value);
                } else if constexpr (sizeof(T) == 4) {
                    return callbacks.MemoryCompareExchange32(location, expected, value);
                } else {
                    return callbacks.MemoryCompareExchange64(location, expected, value);
                }
            };
            REQUIRE(cas(initial, replacement) == initial);
            REQUIRE(memory.Read64(location) == replacement);
            REQUIRE(cas(initial, initial) == replacement);
            REQUIRE(memory.Read64(location) == replacement);
        };
        exercise(base, u8{0x12}, u8{0x23});
        exercise(base, u16{0x1234}, u16{0x2345});
        exercise(base, u32{0x12345678}, u32{0x23456789});
        exercise(end - 8, u64{0x123456789abcdef0}, u64{0x23456789abcdef01});
        REQUIRE_FALSE(callbacks.CheckAtomicAccess(base + 1, 2));
        REQUIRE_FALSE(callbacks.CheckAtomicAccess(base + 2, 4));
        REQUIRE_FALSE(callbacks.CheckAtomicAccess(base + 4, 8));
        REQUIRE_FALSE(callbacks.CheckAtomicAccess(end, 8));
        REQUIRE_FALSE(callbacks.CheckAtomicAccess(1ULL << 32, 8));
    }
    SECTION("kernel permission changes and unmapping revoke atomic access") {
        REQUIRE(memory.IsReadWriteVirtualAddressRange(base, end - base));
        for (const auto attribute :
             {Kernel::KMemoryAttribute::Uncached, Kernel::KMemoryAttribute::None}) {
            REQUIRE(table
                        .SetMemoryAttribute(address, Kernel::PageSize,
                                            Kernel::KMemoryAttribute::Uncached, attribute)
                        .IsSuccess());
            REQUIRE(memory.IsReadWriteVirtualAddressRange(base, end - base));
        }
        for (const auto permission :
             {Kernel::Svc::MemoryPermission::Read, Kernel::Svc::MemoryPermission::None}) {
            REQUIRE(
                table.SetMemoryPermission(address + Kernel::PageSize, Kernel::PageSize, permission)
                    .IsSuccess());
            REQUIRE(memory.IsReadWriteVirtualAddressRange(base, Kernel::PageSize));
            REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base, 2 * Kernel::PageSize));
            REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base + Kernel::PageSize, 1));
            memory.Write64(base + Kernel::PageSize, 0x1234);
            kernel.SetCurrentEmuThread(nullptr);
            REQUIRE(callbacks.MemoryCompareExchange64(base + Kernel::PageSize, 0x1234, 0x5678) ==
                    0x1234);
            REQUIRE(memory.Read64(base + Kernel::PageSize) == 0x1234);
            kernel.SetCurrentEmuThread(&thread);
            REQUIRE(table
                        .SetMemoryPermission(address + Kernel::PageSize, Kernel::PageSize,
                                             Kernel::Svc::MemoryPermission::ReadWrite)
                        .IsSuccess());
            REQUIRE(memory.IsReadWriteVirtualAddressRange(base, end - base));
        }
        REQUIRE(table.SetHeapSize(&address, 2 * Kernel::PageSize).IsSuccess());
        REQUIRE(memory.IsReadWriteVirtualAddressRange(base, 2 * Kernel::PageSize));
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base + 2 * Kernel::PageSize, 1));
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base, end - base));
        REQUIRE(table.SetHeapSize(&address, 4 * Kernel::PageSize).IsSuccess());
        REQUIRE(memory.IsReadWriteVirtualAddressRange(base, end - base));
    }
    SECTION("debugger and rasterizer page tags do not change user permissions") {
        memory.MarkRegionDebug(base, Kernel::PageSize, true);
        REQUIRE(memory.IsReadWriteVirtualAddressRange(base, 8));
        REQUIRE(callbacks.CheckAtomicAccess(base, 8));
        memory.MarkRegionDebug(base, Kernel::PageSize, false);
        memory.RasterizerMarkRegionCached(base, Kernel::PageSize, true);
        REQUIRE(memory.IsReadWriteVirtualAddressRange(base, 8));
        REQUIRE(callbacks.CheckAtomicAccess(base, 8));
        REQUIRE(
            table
                .SetMemoryPermission(address, Kernel::PageSize, Kernel::Svc::MemoryPermission::Read)
                .IsSuccess());
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base, 8));
        memory.RasterizerMarkRegionCached(base, Kernel::PageSize, false);
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base, 8));
    }
    SECTION("empty, overflowing, and out-of-range checks fail closed") {
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base, 0));
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(std::numeric_limits<u64>::max(), 2));
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(0, std::numeric_limits<u64>::max()));
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(1ULL << 32, 1));
        REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(base, (1ULL << 32) - base + 1));
    }
}

TEST_CASE("Atomic permission lookup retains high guest addresses and independent aliases",
          "[core][arm]") {
    const bool multicore = Settings::values.use_multi_core.GetValue();
    Settings::values.use_multi_core = false;
    SCOPE_EXIT {
        Settings::values.use_multi_core = multicore;
    };
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    auto& table = process.GetPageTable().GetImpl();
    auto& memory = process.GetMemory();
    table.Resize(42, 12);
    memory.SetCurrentPageTable(process);
    constexpr u64 low = 0x4000;
    constexpr u64 high = (1ULL << 41) + low;
    constexpr u64 physical = Core::DramMemoryMap::Base + 0x10000;
    memory.MapMemoryRegion(table, low, Kernel::PageSize, physical,
                           Common::MemoryPermission::ReadWrite, false);
    memory.MapMemoryRegion(table, high, Kernel::PageSize, physical, Common::MemoryPermission::Read,
                           false);
    SCOPE_EXIT {
        memory.UnmapRegion(table, low, Kernel::PageSize, false);
        memory.UnmapRegion(table, high, Kernel::PageSize, false);
    };
    REQUIRE(memory.IsReadWriteVirtualAddressRange(low, 8));
    REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(high, 8));
    memory.ProtectRegion(table, high, Kernel::PageSize, Common::MemoryPermission::ReadWrite);
    REQUIRE(memory.IsReadWriteVirtualAddressRange(high, 8));
    memory.ProtectRegion(table, low, Kernel::PageSize, Common::MemoryPermission::Write);
    REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(low, 8));
    REQUIRE(memory.IsReadWriteVirtualAddressRange(high, 8));
    memory.UnmapRegion(table, high, Kernel::PageSize, false);
    REQUIRE_FALSE(memory.IsReadWriteVirtualAddressRange(high, 8));
}
