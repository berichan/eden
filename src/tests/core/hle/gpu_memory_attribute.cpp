// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/hle/kernel/k_memory_block_manager.h"
#include "core/hle/kernel/k_page_group.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_system_resource.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/gsv/gpu_memory_pool.h"
#include "core/hle/service/gsv/graphics_device_table.h"
#include "core/memory.h"

TEST_CASE("GPU sharable attributes survive memory block splitting and independent cache changes",
          "[core][ipc]") {
    using namespace Kernel;
    KDynamicPageManager pages;
    REQUIRE(pages.Initialize(KVirtualAddress{0x100000}, 0x10000, PageSize).IsSuccess());
    KMemoryBlockSlabHeap heap;
    heap.Initialize(&pages, 32);
    KMemoryBlockSlabManager slab;
    slab.Initialize(&pages, &heap);
    KMemoryBlockManager blocks;
    REQUIRE(
        blocks.Initialize(KProcessAddress{0}, KProcessAddress{16 * PageSize}, &slab).IsSuccess());
    SCOPE_EXIT {
        blocks.Finalize(&slab, [](Common::ProcessAddress, u64) {});
    };
    const auto update = [&](size_t start, size_t count, KMemoryAttribute mask,
                            KMemoryAttribute attribute) {
        Result result;
        KMemoryBlockManagerUpdateAllocator allocator{&result, &slab};
        REQUIRE(result.IsSuccess());
        blocks.UpdateAttribute(&allocator, KProcessAddress{start * PageSize}, count, mask,
                               attribute);
        REQUIRE(blocks.CheckState());
    };
    {
        Result result;
        KMemoryBlockManagerUpdateAllocator allocator{&result, &slab};
        REQUIRE(result.IsSuccess());
        blocks.Update(&allocator, KProcessAddress{0}, 16, KMemoryState::Normal,
                      KMemoryPermission::UserReadWrite, KMemoryAttribute::None,
                      KMemoryBlockDisableMergeAttribute::None,
                      KMemoryBlockDisableMergeAttribute::None);
    }
    update(3, 4, KMemoryAttribute::GpuSharable, KMemoryAttribute::GpuSharable);
    update(5, 4, KMemoryAttribute::GpuSharable, KMemoryAttribute::GpuSharable);
    for (size_t page = 0; page < 16; ++page) {
        const auto info = blocks.FindBlock(KProcessAddress{page * PageSize})->GetMemoryInfo();
        REQUIRE(info.GetState() == KMemoryState::Normal);
        REQUIRE(info.GetPermission() == KMemoryPermission::UserReadWrite);
        REQUIRE(
            info.GetSvcMemoryInfo().attribute ==
            (page >= 3 && page < 9 ? Svc::MemoryAttribute::GpuSharable : Svc::MemoryAttribute{}));
        REQUIRE(info.GetIpcLockCount() == 0);
    }
    REQUIRE(blocks.FindBlock(KProcessAddress{3 * PageSize})->GetNumPages() == 6);
    update(4, 2, KMemoryAttribute::GpuSharable, KMemoryAttribute::None);
    REQUIRE(blocks.FindBlock(KProcessAddress{4 * PageSize})->GetAttribute() ==
            KMemoryAttribute::None);
    REQUIRE(blocks.FindBlock(KProcessAddress{3 * PageSize})->GetAttribute() ==
            KMemoryAttribute::GpuSharable);
    REQUIRE(blocks.FindBlock(KProcessAddress{6 * PageSize})->GetAttribute() ==
            KMemoryAttribute::GpuSharable);
    update(3, 1, KMemoryAttribute::Uncached, KMemoryAttribute::Uncached);
    REQUIRE(blocks.FindBlock(KProcessAddress{3 * PageSize})->GetAttribute() ==
            (KMemoryAttribute::GpuSharable | KMemoryAttribute::Uncached));
    update(3, 1, KMemoryAttribute::GpuSharable, KMemoryAttribute::None);
    REQUIRE(blocks.FindBlock(KProcessAddress{3 * PageSize})->GetAttribute() ==
            KMemoryAttribute::Uncached);
    update(3, 1, KMemoryAttribute::Uncached, KMemoryAttribute::None);
    update(6, 3, KMemoryAttribute::GpuSharable, KMemoryAttribute::None);
    REQUIRE(blocks.FindBlock(KProcessAddress{0})->GetNumPages() == 16);
    REQUIRE(blocks.CheckState());
}

TEST_CASE("GPU pins retain real heap pages until partial unlock or process table teardown",
          "[core][ipc]") {
    using namespace Kernel;
    const bool multicore = Settings::values.use_multi_core.GetValue();
    Settings::values.use_multi_core = false;
    SCOPE_EXIT {
        Settings::values.use_multi_core = multicore;
    };
    Core::System system;
    system.Initialize();
    auto& kernel = system.Kernel();
    kernel.Initialize();
    SCOPE_EXIT {
        kernel.Shutdown();
    };
    KProcess process{kernel};
    KThread thread{kernel};
    KAutoObject::Create(&process);
    kernel.SetCurrentEmuThread(&thread);
    REQUIRE(KThread::InitializeDummyThread(system, &thread, &process).IsSuccess());
    SCOPE_EXIT {
        process.UnregisterThread(kernel, &thread);
        process.Close(kernel);
        kernel.UnregisterKernelObject(&thread);
        kernel.UnregisterKernelObject(&process);
        kernel.SetCurrentEmuThread(nullptr);
    };
    auto& table = process.GetPageTable();
    REQUIRE(table
                .Initialize(Svc::CreateProcessFlag::AddressSpace32Bit, false, false, false,
                            KMemoryManager::Pool::Application, 0x200000, PageSize,
                            &kernel.GetAppSystemResource(), kernel.GetSystemResourceLimit(),
                            process.GetMemory(), 0)
                .IsSuccess());
    process.GetMemory().SetCurrentPageTable(process);
    bool finalized{};
    SCOPE_EXIT {
        if (!finalized)
            table.Finalize();
    };
    const size_t free_before = kernel.MemoryManager().GetFreeSize();
    REQUIRE(table.SetMaxHeapSize(4 * PageSize).IsSuccess());
    KProcessAddress address;
    REQUIRE(table.SetHeapSize(&address, 4 * PageSize).IsSuccess());
    REQUIRE(kernel.MemoryManager().GetFreeSize() == free_before - 4 * PageSize);
    SECTION("direct pins and teardown") {
        Svc::lp64::PhysicalMemoryInfo info{1, 2, 3};
        REQUIRE(table.LockGpuSharableMemory(&info, address, PageSize) ==
                ResultInvalidCurrentMemory);
        REQUIRE(info.physical_address == 1);
        REQUIRE(info.virtual_address == 2);
        REQUIRE(info.size == 3);
        REQUIRE(table
                    .SetGpuMemoryAttribute(address, 4 * PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        REQUIRE(table.LockGpuSharableMemory(&info, address, 2 * PageSize).IsSuccess());
        REQUIRE(info.virtual_address == GetInteger(address));
        REQUIRE(info.size > 0);
        REQUIRE(info.size <= 2 * PageSize);
        const size_t pinned_size = info.size;
        REQUIRE(table
                    .SetMemoryAttribute(address, PageSize, KMemoryAttribute::Uncached,
                                        KMemoryAttribute::Uncached)
                    .IsSuccess());
        KMemoryInfo cached_info;
        Svc::PageInfo cached_page;
        REQUIRE(table.QueryInfo(&cached_info, &cached_page, address).IsSuccess());
        REQUIRE(cached_info.GetAttribute() ==
                (KMemoryAttribute::GpuSharable | KMemoryAttribute::GpuShared |
                 KMemoryAttribute::Uncached));
        REQUIRE(table
                    .SetMemoryAttribute(address, PageSize, KMemoryAttribute::Uncached,
                                        KMemoryAttribute::None)
                    .IsSuccess());
        REQUIRE(table.LockGpuSharableMemory(&info, address, PageSize) ==
                ResultInvalidCurrentMemory);
        REQUIRE(table.SetGpuMemoryAttribute(address, PageSize, KMemoryAttribute::GpuSharable,
                                            KMemoryAttribute::None) == ResultInvalidCurrentMemory);
        REQUIRE(table.SetMemoryPermission(address, PageSize, Svc::MemoryPermission::Read) ==
                ResultInvalidCurrentMemory);
        REQUIRE(table.UnlockGpuSharableMemory(address, 4 * PageSize) == ResultInvalidCurrentMemory);
        REQUIRE(table.UnlockGpuSharableMemory(address, PageSize).IsSuccess());
        REQUIRE(table.UnlockGpuSharableMemory(address, PageSize) == ResultInvalidCurrentMemory);
        KMemoryInfo memory_info;
        Svc::PageInfo page_info;
        REQUIRE(table.QueryInfo(&memory_info, &page_info, address).IsSuccess());
        REQUIRE(memory_info.GetAttribute() == KMemoryAttribute::GpuSharable);
        if (pinned_size > PageSize) {
            REQUIRE(table.QueryInfo(&memory_info, &page_info, address + PageSize).IsSuccess());
            REQUIRE(memory_info.GetAttribute() ==
                    (KMemoryAttribute::GpuSharable | KMemoryAttribute::GpuShared));
        }
        REQUIRE(table.LockGpuSharableMemory(&info, address, PageSize).IsSuccess());
    }
    SECTION("registration rollback, device lifetime and explicit release") {
        using namespace Service::GSV;
        auto devices = std::make_shared<GraphicsDeviceTable>();
        u8 device_id{};
        REQUIRE(devices->Create(&device_id, 1, 0, 0).IsSuccess());
        GpuMemoryPool pool{kernel, &process, devices};
        REQUIRE(table
                    .SetGpuMemoryAttribute(address, PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        u32 handle{0xffffffff};
        REQUIRE(pool.Register(&handle, GetInteger(address), GetInteger(address) + 4 * PageSize - 1,
                              0, device_id)
                    .IsError());
        REQUIRE(handle == 0xffffffff);
        REQUIRE(devices->Get(device_id)->memory_references == 0);
        KMemoryInfo info;
        Svc::PageInfo page;
        REQUIRE(table.QueryInfo(&info, &page, address).IsSuccess());
        REQUIRE(info.GetAttribute() == KMemoryAttribute::GpuSharable);
        REQUIRE(pool.Register(&handle, GetInteger(address), GetInteger(address) + PageSize - 1, 0,
                              device_id)
                    .IsSuccess());
        REQUIRE(handle == 0x01000001);
        REQUIRE(devices->Destroy(device_id) == GraphicsDeviceTable::InvalidDevice);
        REQUIRE(pool.Unregister(handle).IsSuccess());
        REQUIRE(pool.Unregister(handle) == Result{0x280f3});
        REQUIRE(devices->Destroy(device_id).IsSuccess());
        REQUIRE(pool.Register(&handle, GetInteger(address), GetInteger(address) + PageSize - 1, 0,
                              device_id) == GraphicsDeviceTable::InvalidDevice);
        REQUIRE(devices->Create(&device_id, 2, 0, 0).IsSuccess());
        {
            GpuMemoryPool closing{kernel, &process, devices};
            REQUIRE(closing
                        .Register(&handle, GetInteger(address), GetInteger(address) + PageSize - 1,
                                  0, device_id)
                        .IsSuccess());
            REQUIRE(devices->Destroy(device_id) == GraphicsDeviceTable::InvalidDevice);
        }
        REQUIRE(devices->Get(device_id)->memory_references == 0);
        REQUIRE(table.QueryInfo(&info, &page, address).IsSuccess());
        REQUIRE(info.GetAttribute() == KMemoryAttribute::GpuSharable);
        REQUIRE(devices->Destroy(device_id).IsSuccess());
    }
    SECTION("fragmented backing returns and pins only its physical prefix") {
        const auto physical = kernel.MemoryManager().AllocateAndOpenContinuous(
            4, 1,
            KMemoryManager::EncodeOption(KMemoryManager::Pool::Application,
                                         KMemoryManager::Direction::FromFront));
        REQUIRE(physical != 0);
        SCOPE_EXIT {
            kernel.MemoryManager().Close(physical, 4);
        };
        KPageGroup group{kernel, &kernel.GetAppSystemResource().GetBlockInfoManager()};
        REQUIRE(group.AddBlock(physical, 2).IsSuccess());
        REQUIRE(group.AddBlock(physical + 3 * PageSize, 1).IsSuccess());
        const auto fragmented = address + 16 * PageSize;
        REQUIRE(table
                    .MapPageGroup(fragmented, group, KMemoryState::Normal,
                                  KMemoryPermission::UserReadWrite)
                    .IsSuccess());
        REQUIRE(table
                    .SetGpuMemoryAttribute(fragmented, 3 * PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        Svc::lp64::PhysicalMemoryInfo info;
        REQUIRE(table.LockGpuSharableMemory(&info, fragmented, 3 * PageSize).IsSuccess());
        REQUIRE(info.physical_address == GetInteger(physical));
        REQUIRE(info.virtual_address == GetInteger(fragmented));
        REQUIRE(info.size == 2 * PageSize);
        REQUIRE(table.UnlockGpuSharableMemory(fragmented + PageSize, PageSize).IsSuccess());
        REQUIRE(
            table.LockGpuSharableMemory(&info, fragmented + 2 * PageSize, PageSize).IsSuccess());
        REQUIRE(info.physical_address == GetInteger(physical) + 3 * PageSize);
        REQUIRE(info.size == PageSize);
        REQUIRE(table.UnlockGpuSharableMemory(fragmented, 3 * PageSize) ==
                ResultInvalidCurrentMemory);
        REQUIRE(table.UnlockGpuSharableMemory(fragmented, PageSize).IsSuccess());
        // Leave the last pin for finalization, with a gap between it and the earlier allocation.
    }
    table.Finalize();
    finalized = true;
    REQUIRE(kernel.MemoryManager().GetFreeSize() == free_before);
}
