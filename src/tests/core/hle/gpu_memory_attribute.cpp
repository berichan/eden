// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_memory_block_manager.h"
#include "core/hle/kernel/k_page_group.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_system_resource.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/gsv/gpu_address_space_pool.h"
#include "core/hle/service/gsv/gpu_memory_pool.h"
#include "core/hle/service/gsv/graphics_device_table.h"
#include "core/hle/service/gsv/native_image_pool.h"
#include "core/hle/service/gsv/native_memory_descriptions.h"
#include "core/memory.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/shader_environment.h"

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

TEST_CASE("Device translations preserve 42-bit CPU addresses and physical RAM above eight GiB",
          "[core][ipc]") {
    const auto layout = Settings::values.memory_layout_mode.GetValue();
    const auto multicore = Settings::values.use_multi_core.GetValue();
    Settings::values.memory_layout_mode = Settings::MemoryLayout::Memory_12Gb;
    Settings::values.use_multi_core = false;
    SCOPE_EXIT {
        Settings::values.memory_layout_mode = layout;
        Settings::values.use_multi_core = multicore;
    };
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(42, 12);
    process.GetMemory().SetCurrentPageTable(process);
    constexpr u64 cpu = (1ULL << 41) + 0x1000;
    constexpr u64 physical_offset = 10ULL << 30;
    process.GetMemory().MapMemoryRegion(table, cpu, Kernel::PageSize,
                                        Core::DramMemoryMap::Base + physical_offset,
                                        Common::MemoryPermission::ReadWrite, false);
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Tegra::MaxwellDeviceMemoryManager backing{system.DeviceMemory()};
    backing.BindInterface(&rasterizer);
    const auto first_asid = backing.RegisterProcess(&process.GetMemory());
    const auto asid = backing.RegisterProcess(&process.GetMemory());
    SCOPE_EXIT {
        backing.UnregisterProcess(asid);
        backing.UnregisterProcess(first_asid);
        process.GetMemory().UnmapRegion(table, cpu, Kernel::PageSize, false);
    };
    const auto device = backing.Allocate(Kernel::PageSize);
    REQUIRE(device != 0);
    SCOPE_EXIT {
        backing.Unmap(device, Kernel::PageSize);
        backing.Free(device, Kernel::PageSize);
    };
    backing.Map(device, cpu, Kernel::PageSize, asid, true);
    REQUIRE(backing.GetPhysicalRawAddressFromDAddr(device) == physical_offset);
    backing.Write<u32>(device, 0x1234abcd);
    REQUIRE(process.GetMemory().Read32(cpu) == 0x1234abcd);
    backing.UpdatePagesCachedCount(device, Kernel::PageSize, 1);
    REQUIRE(std::get<1>(table.entries[cpu >> 12].PointerTypeBlock(true)) ==
            Common::PageType::RasterizerCachedMemory);
    backing.UpdatePagesCachedCount(device, Kernel::PageSize, -1);
    REQUIRE(std::get<1>(table.entries[cpu >> 12].PointerTypeBlock(true)) ==
            Common::PageType::Memory);
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
    SECTION("native descriptions retain registered slices until their final release") {
        using namespace Service::GSV;
        auto devices = std::make_shared<GraphicsDeviceTable>();
        u8 device{};
        REQUIRE(devices->Create(&device, 1, 0, 0).IsSuccess());
        auto pool = std::make_shared<GpuMemoryPool>(kernel, &process, devices);
        REQUIRE(table
                    .SetGpuMemoryAttribute(address, 4 * PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        u32 memory{};
        const u64 base = GetInteger(address);
        REQUIRE(pool->Register(&memory, base, base + 4 * PageSize - 1, 0, device).IsSuccess());
        REQUIRE(pool->RetainReference(0, base, PageSize, 0) == Result{0x280f3});
        REQUIRE(pool->ReleaseReference(memory) == Result{0x280f3});
        NativeMemoryDescriptions descriptions;
        u32 description{}, other_type{};
        REQUIRE(descriptions.Create(&description, 3, PageSize).IsSuccess());
        REQUIRE(descriptions.Create(&other_type, 1, PageSize).IsSuccess());
        REQUIRE(descriptions.Bind(0, pool, base, PageSize, memory, 0) == Result{0x280f3});
        REQUIRE(descriptions.Bind(other_type, pool, base, PageSize, memory, 0) == Result{0x100f3});
        REQUIRE(descriptions.Bind(description, pool, base, 0, memory, 0) == Result{0x100f3});
        REQUIRE(descriptions.Bind(description, pool, base, 2 * PageSize, memory, 0) ==
                Result{0x100f3});
        REQUIRE(descriptions.Bind(description, pool, base, PageSize / 2, memory, 0) ==
                ResultNotImplemented);
        REQUIRE(descriptions.Bind(description, pool, base, PageSize, 0, 0) == Result{0x280f3});
        REQUIRE(descriptions.Bind(description, pool, base + PageSize, PageSize, memory, 0) ==
                Result{0x100f3});
        REQUIRE(descriptions.Bind(description, pool, base, PageSize, memory, ~u64{0}) ==
                Result{0x100f3});
        REQUIRE(descriptions.Bind(description, pool, base + 4 * PageSize, PageSize, memory,
                                  4 * PageSize) == Result{0x100f3});
        REQUIRE_FALSE(descriptions.Get(description)->binding);
        REQUIRE(descriptions.Bind(description, pool, base + PageSize, PageSize, memory, PageSize)
                    .IsSuccess());
        const auto binding = descriptions.Get(description)->binding;
        REQUIRE(binding.has_value());
        REQUIRE(binding->address == base + PageSize);
        REQUIRE(binding->offset == PageSize);
        REQUIRE(binding->memory_handle == memory);
        REQUIRE(descriptions.Bind(description, pool, base, PageSize, memory, 0) == Result{0x280f3});
        auto* event = KEvent::Create(kernel);
        REQUIRE(event != nullptr);
        event->Initialize(kernel, nullptr);
        KEvent::Register(kernel, event);
        SCOPE_EXIT {
            event->GetReadableEvent().Close(kernel);
            event->Close(kernel);
        };
        {
            NativeMemoryDescriptions closing;
            u32 second{};
            REQUIRE(closing.Create(&second, 3, PageSize).IsSuccess());
            REQUIRE(closing.Bind(second, pool, base + 2 * PageSize, PageSize, memory, 2 * PageSize)
                        .IsSuccess());
            REQUIRE(pool->Unregister(memory, event).IsSuccess());
            REQUIRE(pool->RetainReference(memory, base, PageSize, 0) == Result{0x280f3});
            REQUIRE_FALSE(event->GetReadableEvent().IsSignaled(kernel));
            REQUIRE(descriptions.Destroy(description).IsSuccess());
            REQUIRE_FALSE(event->GetReadableEvent().IsSignaled(kernel));
            REQUIRE(devices->Destroy(device) == GraphicsDeviceTable::InvalidDevice);
        }
        REQUIRE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(pool->ReleaseReference(memory) == Result{0x280f3});
        KMemoryInfo info;
        Svc::PageInfo page;
        REQUIRE(table.QueryInfo(&info, &page, address).IsSuccess());
        REQUIRE(info.GetAttribute() == KMemoryAttribute::GpuSharable);
        REQUIRE(devices->Destroy(device).IsSuccess());
    }
    SECTION("native images validate real backing and keep it alive through external consumers") {
        using namespace Service::GSV;
        auto devices = std::make_shared<GraphicsDeviceTable>();
        u8 device{};
        REQUIRE(devices->Create(&device, 1, 0, 0).IsSuccess());
        auto pool = std::make_shared<GpuMemoryPool>(kernel, &process, devices);
        REQUIRE(table
                    .SetGpuMemoryAttribute(address, 4 * PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        u32 memory{};
        const u64 base = GetInteger(address);
        REQUIRE(pool->Register(&memory, base, base + 4 * PageSize - 1, 0, device).IsSuccess());
        auto descriptions = std::make_shared<NativeMemoryDescriptions>();
        u32 description{}, small{};
        REQUIRE(descriptions->Create(&description, 3, PageSize).IsSuccess());
        REQUIRE(descriptions->Create(&small, 3, 256).IsSuccess());
        NativeImagePreference preference{};
        preference.type = 1;
        preference.row_alignment_log2 = 3;
        preference.format = 1;
        preference.options = 1;
        std::shared_ptr<const NativeImage> consumer;
        {
            NativeImagePool images{descriptions};
            u32 handle{0xfeed};
            REQUIRE(images.Create(&handle, preference, 1, 0) ==
                    NativeMemoryDescriptions::InvalidHandle);
            REQUIRE(handle == 0xfeed);
            REQUIRE(images.Create(&handle, preference, 1, description) ==
                    NativeMemoryDescriptions::Busy);
            REQUIRE(handle == 0xfeed);
            REQUIRE(descriptions->Bind(small, pool, base, 256, memory, 0).IsSuccess());
            REQUIRE(images.Create(&handle, preference, 1, small) == NativeMemoryDescriptions::Busy);
            REQUIRE(handle == 0xfeed);
            REQUIRE(descriptions->Destroy(small).IsSuccess());
            REQUIRE(
                descriptions->Bind(description, pool, base + PageSize, PageSize, memory, PageSize)
                    .IsSuccess());
            REQUIRE(images.Create(&handle, preference, 0, description) ==
                    NativeMemoryDescriptions::InvalidArgument);
            REQUIRE(images.Create(&handle, preference, 2, description) == ResultNotImplemented);
            auto invalid = preference;
            invalid.packed_extent = 255;
            invalid.pitch_gobs = 1;
            REQUIRE(images.Create(&handle, invalid, 1, description) ==
                    NativeMemoryDescriptions::InvalidArgument);
            REQUIRE(handle == 0xfeed);
            REQUIRE(images.Create(&handle, preference, 1, description).IsSuccess());
            REQUIRE(handle == 1);
            REQUIRE_FALSE(images.AcquireAlias(handle));
            REQUIRE(images.RegisterAlias(0, 1) == NativeMemoryDescriptions::InvalidHandle);
            for (const u32 count : {0U, 2U, 0xffffffffU}) {
                REQUIRE(images.RegisterAlias(handle, count) ==
                        NativeMemoryDescriptions::InvalidArgument);
            }
            REQUIRE(images.UnregisterAlias(handle) == NativeMemoryDescriptions::InvalidHandle);
            for (size_t i = 0; i < 255; ++i) {
                REQUIRE(images.RegisterAlias(handle, 1).IsSuccess());
            }
            REQUIRE(images.RegisterAlias(handle, 1) == NativeMemoryDescriptions::OutOfMemory);
            REQUIRE(images.Destroy(handle) == NativeMemoryDescriptions::Busy);
            for (size_t i = 0; i < 254; ++i) {
                REQUIRE(images.UnregisterAlias(handle).IsSuccess());
            }
            auto alias = images.AcquireAlias(handle);
            REQUIRE(alias != nullptr);
            REQUIRE(images.UnregisterAlias(handle).IsSuccess());
            REQUIRE_FALSE(images.AcquireAlias(handle));
            REQUIRE(images.Destroy(handle) == NativeMemoryDescriptions::Busy);
            alias.reset();
            REQUIRE(descriptions->Destroy(description) == NativeMemoryDescriptions::Busy);
            u32 duplicate{0xbeef};
            REQUIRE(images.Create(&duplicate, preference, 1, description) ==
                    NativeMemoryDescriptions::Busy);
            REQUIRE(duplicate == 0xbeef);
            REQUIRE_FALSE(images.Acquire(0));
            REQUIRE(images.Destroy(0) == NativeMemoryDescriptions::InvalidHandle);
            consumer = images.Acquire(handle);
            REQUIRE(consumer != nullptr);
            REQUIRE(consumer->layout.size == 512);
            REQUIRE(consumer->layout.row_pitch == 64);
            REQUIRE(consumer->preference.format == 1);
            REQUIRE(consumer->Memory().binding->address == base + PageSize);
            REQUIRE(consumer->Memory().binding->offset == PageSize);
            REQUIRE(consumer->Memory().binding->memory_handle == memory);
            REQUIRE(images.Destroy(handle) == NativeMemoryDescriptions::Busy);
            consumer.reset();
            REQUIRE(images.Destroy(handle).IsSuccess());
            REQUIRE_FALSE(images.Acquire(handle));
            REQUIRE(images.Destroy(handle) == NativeMemoryDescriptions::InvalidHandle);
            REQUIRE(images.Create(&handle, preference, 1, description).IsSuccess());
            REQUIRE(handle == 2);
            REQUIRE(images.RegisterAlias(handle, 1).IsSuccess());
            consumer = images.AcquireAlias(handle);
            images.ClearAliases();
            REQUIRE_FALSE(images.AcquireAlias(handle));
        }
        REQUIRE(consumer->Memory().binding->address == base + PageSize);
        REQUIRE(descriptions->Destroy(description) == NativeMemoryDescriptions::Busy);
        auto* event = KEvent::Create(kernel);
        REQUIRE(event != nullptr);
        event->Initialize(kernel, nullptr);
        KEvent::Register(kernel, event);
        SCOPE_EXIT {
            event->GetReadableEvent().Close(kernel);
            event->Close(kernel);
        };
        REQUIRE(pool->Unregister(memory, event).IsSuccess());
        std::weak_ptr<GpuMemoryPool> retained_pool = pool;
        std::weak_ptr<NativeMemoryDescriptions> retained_descriptions = descriptions;
        pool.reset();
        descriptions.reset();
        REQUIRE_FALSE(retained_pool.expired());
        REQUIRE_FALSE(retained_descriptions.expired());
        REQUIRE_FALSE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(devices->Destroy(device) == GraphicsDeviceTable::InvalidDevice);
        consumer.reset();
        REQUIRE(retained_pool.expired());
        REQUIRE(retained_descriptions.expired());
        REQUIRE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(devices->Destroy(device).IsSuccess());
    }
    SECTION("GPU mappings share the retained CPU pages and delay release completion") {
        using namespace Service::GSV;
        Tegra::GPU gpu{system, false, false};
        Null::RasterizerNull rasterizer{gpu};
        Tegra::MaxwellDeviceMemoryManager backing{system.DeviceMemory()};
        backing.BindInterface(&rasterizer);
        GpuAddressSpacePool spaces;
        auto manager =
            std::make_unique<Tegra::MemoryManager>(system, backing, 40, 1ULL << 40, 16, 12);
        manager->BindRasterizer(&rasterizer);
        auto* view = manager.get();
        u32 space{};
        REQUIRE(spaces.Add(&space, std::move(manager), 0).IsSuccess());
        auto devices = std::make_shared<GraphicsDeviceTable>();
        u8 device{};
        REQUIRE(devices->Create(&device, 1, 0, 0).IsSuccess());
        GpuMemoryPool pool{kernel, &process, devices};
        REQUIRE(table
                    .SetGpuMemoryAttribute(address, 4 * PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        u32 handle{};
        REQUIRE(pool.Register(&handle, GetInteger(address), GetInteger(address) + 4 * PageSize - 1,
                              0, device)
                    .IsSuccess());
        constexpr u64 va = 0x8000000;
        constexpr u64 size = 2 * PageSize;
        const GpuMemoryPool::MapOptions options{0, 0, 6, 6, 0, device, 1};
        u8 selected{0xff};
        process.GetMemory().Write32(address + PageSize, 0x12345678);
        REQUIRE(pool.Map(&selected, spaces, backing, space, va, va + size - 1, handle, PageSize,
                         options)
                    .IsSuccess());
        REQUIRE(selected == 6);
        REQUIRE(view->Read<u32>(va) == 0x12345678);
        view->Write<u32>(va + PageSize, 0xabcdef01);
        REQUIRE(process.GetMemory().Read32(address + 2 * PageSize) == 0xabcdef01);
        REQUIRE(view->GetPageKind(va) == Tegra::PTEKind::GENERIC_16BX2);
        REQUIRE(backing.GetPhysicalRawAddressFromDAddr(*view->GpuToCpuAddress(va)) ==
                system.DeviceMemory().GetRawPhysicalAddr(
                    process.GetMemory().GetPointerSilent(address + PageSize)));
        auto alias_manager =
            std::make_unique<Tegra::MemoryManager>(system, backing, 40, 1ULL << 40, 16, 12);
        alias_manager->BindRasterizer(&rasterizer);
        auto* alias_view = alias_manager.get();
        u32 alias_space{};
        REQUIRE(spaces.Add(&alias_space, std::move(alias_manager), 0).IsSuccess());
        REQUIRE(pool.Map(&selected, spaces, backing, alias_space, va, va + size - 1, handle,
                         PageSize, options)
                    .IsSuccess());
        REQUIRE(alias_view->GpuToCpuAddress(va) == view->GpuToCpuAddress(va));
        REQUIRE(alias_view->Read<u32>(va + PageSize) == 0xabcdef01);

        {
            std::array<u8, 8> transfer;
            transfer.fill(0xcc);
            REQUIRE_FALSE(view->ReadBlockChecked(va + size - 4, transfer));
            REQUIRE(transfer.front() == 0xcc);
            const auto original = view->Read<u32>(va + size - 4);
            REQUIRE_FALSE(view->WriteBlockChecked(va + size - 4, transfer));
            REQUIRE(view->Read<u32>(va + size - 4) == original);
            using Instruction = Shader::Ampere::Instruction;
            const std::array native{
                Instruction{{0x00077802, 13, 0x00000f00, 0}},
                Instruction{{0xff007388, 7, 0x00000800, 0}},
                Instruction{{0xff007984, 0, 0x00000800, 0}},
                Instruction{{0xff007388, 0x400, 0x00000800, 0}},
                Instruction{{0x0000794d, 0, 0x03800000, 0}},
            };
            constexpr u32 code_offset = 0x200;
            REQUIRE(view->WriteBlockChecked(
                va + code_offset, {reinterpret_cast<const u8*>(native.data()), sizeof(native)}));
            REQUIRE(process.GetMemory().Read32(address + PageSize + code_offset) ==
                    native[0].words[0]);
            constexpr u32 constant = 0x76543210;
            REQUIRE(view->WriteBlockChecked(
                va + 0x400, {reinterpret_cast<const u8*>(&constant), sizeof(constant)}));
            std::array<u32, 64> descriptor{};
            descriptor[12] = descriptor[13] = descriptor[14] = 1;
            descriptor[17] = 128;
            descriptor[18] = (32U << 16) | 0x30;
            descriptor[19] = (1U << 16) | 1;
            descriptor[20] = 1;
            descriptor[23] = 32;
            descriptor[50] = 16;
            descriptor[48] = static_cast<u32>(va + code_offset);
            descriptor[32] = static_cast<u32>(va + 0x400);
            descriptor[33] = 1U << 19; // A sixteen-byte constant buffer.
            Tegra::Engines::KeplerCompute compute{*view};
            compute.BindRasterizer(&rasterizer);
            REQUIRE_FALSE(compute.IsNativeMode());
            REQUIRE(compute.ConfigureNativeLaunch(descriptor));
            VideoCommon::ComputeEnvironment env{compute, *view, va + code_offset, 0};
            REQUIRE(env.StartAddress() == 0x50);
            REQUIRE(env.SourceSizeBytes() == sizeof(native));
            REQUIRE(env.CachedSizeBytes() > env.SourceSizeBytes());
            REQUIRE(env.WorkgroupSize() == std::array<u32, 3>{32, 1, 1});
            REQUIRE(env.LocalMemorySize() == 48);
            REQUIRE(env.ReadCbufValue(0, 0) == constant);
            REQUIRE_THROWS_AS(env.ReadCbufValue(8, 0), Shader::RuntimeError);
            REQUIRE_THROWS_AS(env.ReadCbufValue(1, 0), Shader::RuntimeError);
            REQUIRE_THROWS_AS(env.ReadTextureType(0), Shader::RuntimeError);
            REQUIRE(env.Analyze() == env.CalculateHash());
            REQUIRE(env.ReadCbufValue(0, 12) == 0);
            REQUIRE(env.ReadCbufValue(0, 16) == 0);
            for (u32 tail : {13U, 14U, 15U}) {
                REQUIRE_THROWS_AS(env.ReadCbufValue(0, tail), Shader::RuntimeError);
            }
            std::array<u8, 16> storage_descriptor;
            REQUIRE(compute.ReadNativeConstantBuffer(0, 0, storage_descriptor));
            REQUIRE_FALSE(compute.ReadNativeConstantBuffer(0, 1, storage_descriptor));
            REQUIRE_FALSE(compute.ReadNativeConstantBuffer(0, 0xffffffff, storage_descriptor));
            REQUIRE_FALSE(compute.ReadNativeConstantBuffer(8, 0, storage_descriptor));
            auto specialization = descriptor;
            specialization[23] += 16;
            REQUIRE(compute.ConfigureNativeLaunch(specialization));
            VideoCommon::ComputeEnvironment local_variant{compute, *view, va + code_offset, 0};
            REQUIRE(local_variant.CalculateHash() != env.CalculateHash());
            specialization = descriptor;
            specialization[20] |= 2;
            specialization[34] = static_cast<u32>(va + 0x480);
            specialization[35] = 1U << 19;
            REQUIRE(compute.ConfigureNativeLaunch(specialization));
            process.GetMemory().Write32(address + PageSize + 0x480, 11);
            VideoCommon::ComputeEnvironment constants_a{compute, *view, va + code_offset, 0};
            process.GetMemory().Write32(address + PageSize + 0x480, 22);
            VideoCommon::ComputeEnvironment constants_b{compute, *view, va + code_offset, 0};
            REQUIRE(constants_a.CalculateHash() != constants_b.CalculateHash());
            REQUIRE(constants_a.ReadCbufValue(1, 0) == 11);
            REQUIRE(constants_b.ReadCbufValue(1, 0) == 22);
            REQUIRE_THROWS_AS(constants_a.ReadCbufValue(1, 13), Shader::RuntimeError);
            REQUIRE(compute.ConfigureNativeLaunch(descriptor));
            env.SetCachedSize(env.CachedSizeBytes());
            REQUIRE_THROWS_AS(env.SetCachedSize(env.CachedSizeBytes() + 8), Shader::RuntimeError);
            REQUIRE_THROWS_AS(env.ReadInstruction(static_cast<u32>(env.CachedSizeBytes())),
                              Shader::RuntimeError);
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
            Shader::ObjectPool<Shader::IR::Inst> inst_pool;
            Shader::ObjectPool<Shader::IR::Block> block_pool;
            Shader::Maxwell::Flow::CFG cfg{env, flow_pool, env.StartAddress()};
            Shader::HostTranslateInfo host;
            const auto program =
                Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
            u32 shared_reads{}, shared_writes{};
            for (const auto* ir_block : program.blocks) {
                for (const auto& inst : ir_block->Instructions()) {
                    shared_reads += inst.GetOpcode() == Shader::IR::Opcode::LoadSharedU32;
                    shared_writes += inst.GetOpcode() == Shader::IR::Opcode::WriteSharedU32;
                }
            }
            REQUIRE(shared_reads == 1);
            REQUIRE(shared_writes == 2);
            std::stringstream stream{std::ios::in | std::ios::out | std::ios::binary};
            env.Serialize(stream);
            stream.seekg(0);
            VideoCommon::FileEnvironment cached;
            cached.Deserialize(stream);
            REQUIRE(stream.good());
            REQUIRE(cached.StartAddress() == env.StartAddress());
            REQUIRE(cached.WorkgroupSize() == env.WorkgroupSize());
            REQUIRE(cached.ReadCbufValue(0, 0) == constant);
            REQUIRE(cached.ReadInstruction(0x50) == env.ReadInstruction(0x50));
            REQUIRE(cached.TranslateProgramAddress(16) == env.TranslateProgramAddress(16));
            REQUIRE_THROWS_AS(cached.TranslateProgramAddress(1), Shader::RuntimeError);
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> cached_flow_pool;
            Shader::Maxwell::Flow::CFG cached_cfg{cached, cached_flow_pool, cached.StartAddress()};
            REQUIRE(cached_cfg.Functions().size() == cfg.Functions().size());
            // A null backend must not report a nonempty native dispatch as completed work.
            REQUIRE_FALSE(compute.DispatchNativeCompute(descriptor));
            auto invalid = descriptor;
            invalid[48] = static_cast<u32>(va + size - 8);
            REQUIRE(compute.ConfigureNativeLaunch(invalid));
            REQUIRE_THROWS_AS((VideoCommon::ComputeEnvironment{compute, *view, va + size - 8, 0}),
                              Shader::RuntimeError);
            invalid = descriptor;
            invalid[32] = static_cast<u32>(va + size);
            REQUIRE(compute.ConfigureNativeLaunch(invalid));
            VideoCommon::ComputeEnvironment unmapped_constant{compute, *view, va + code_offset, 0};
            REQUIRE_THROWS_AS(unmapped_constant.ReadCbufValue(0, 0), Shader::RuntimeError);
            REQUIRE_FALSE(compute.DispatchNativeCompute(invalid));
            Tegra::MemoryManager split{system, backing, 40, 1ULL << 40, 16, 12};
            split.BindRasterizer(&rasterizer);
            constexpr GPUVAddr split_va = 0x40000000;
            REQUIRE(split.Map(split_va, *view->GpuToCpuAddress(va + PageSize), PageSize,
                              Tegra::PTEKind::PITCH, false) == split_va);
            REQUIRE(split.Map(split_va + PageSize, *view->GpuToCpuAddress(va), PageSize,
                              Tegra::PTEKind::PITCH, false) == split_va + PageSize);
            const GPUVAddr split_code = split_va + PageSize - 32;
            REQUIRE_FALSE(split.IsContinuousRange(split_code, sizeof(native)));
            std::array<u8, sizeof(native)> saved_source;
            REQUIRE(split.ReadBlockChecked(split_code, saved_source));
            REQUIRE(split.WriteBlockChecked(
                split_code, {reinterpret_cast<const u8*>(native.data()), sizeof(native)}));
            auto split_descriptor = descriptor;
            split_descriptor[48] = static_cast<u32>(split_code);
            split_descriptor[20] = 0;
            Tegra::Engines::KeplerCompute split_compute{split};
            REQUIRE(split_compute.ConfigureNativeLaunch(split_descriptor));
            VideoCommon::ComputeEnvironment split_a{split_compute, split, split_code, 0};
            REQUIRE(split_a.SourceSizeBytes() == sizeof(native));
            REQUIRE(split_a.CalculateHash() != 0);
            const u32 changed_offset = 0x800;
            REQUIRE(split.WriteBlockChecked(
                split_code + 3 * sizeof(Instruction) + sizeof(u32),
                {reinterpret_cast<const u8*>(&changed_offset), sizeof(changed_offset)}));
            VideoCommon::ComputeEnvironment split_b{split_compute, split, split_code, 0};
            REQUIRE(split_b.CalculateHash() != split_a.CalculateHash());
            REQUIRE(split.WriteBlockChecked(split_code, saved_source));
            using GraphicsContext = Tegra::Control::NativeGpuGraphicsContext;
            using Qmd = Tegra::Control::NativeQmdScheduler;
            struct ReleaseHost {
                Tegra::MemoryManager& memory;
                Tegra::Engines::KeplerCompute& compute;
                u32 releases{};
                GraphicsContext::QmdBackend Bind() {
                    return {
                        this,
                        [](void*, const Qmd::Release& release) { return release.SizeBytes() == 4; },
                        [](void* context, const Qmd::Descriptor& descriptor) {
                            return static_cast<ReleaseHost*>(context)
                                ->compute.DispatchNativeCompute(descriptor);
                        },
                        [](void* context, const Qmd::Release& release) {
                            auto& self = *static_cast<ReleaseHost*>(context);
                            const u32 value = static_cast<u32>(release.payload);
                            if (!self.memory.WriteBlockChecked(
                                    release.address,
                                    {reinterpret_cast<const u8*>(&value), sizeof(value)}))
                                return false;
                            ++self.releases;
                            return true;
                        }};
                }
            } release_host{*view, compute};
            const auto graphics_state = std::make_shared<Tegra::Control::AmpereGraphicsState>();
            GraphicsContext rejected_release{graphics_state, spaces.Acquire(space)};
            auto qmd = descriptor;
            qmd[12] = 0;
            qmd[24] = static_cast<u32>(va + 0x900);
            qmd[25] = (1U << 30) | (1U << 23);
            qmd[26] = 99;
            qmd[28] = static_cast<u32>(va + size);
            qmd[29] = qmd[25];
            const auto write_qmd = [&] {
                return view->WriteBlockChecked(
                    va + 0x800, {reinterpret_cast<const u8*>(qmd.data()), sizeof(qmd)});
            };
            view->Write<u32>(va + 0x900, 77);
            REQUIRE(write_qmd());
            REQUIRE(rejected_release.SignalQmd(va + 0x800, 3, release_host.Bind()).error ==
                    Qmd::Error::InvalidRelease);
            REQUIRE(view->Read<u32>(va + 0x900) == 77);
            REQUIRE(release_host.releases == 0);
            GraphicsContext accepted_release{graphics_state, spaces.Acquire(space)};
            qmd[29] = 0;
            REQUIRE(write_qmd());
            const auto empty_qmd = accepted_release.SignalQmd(va + 0x800, 3, release_host.Bind());
            REQUIRE(empty_qmd.Succeeded());
            REQUIRE(empty_qmd.dispatches == 0);
            REQUIRE(empty_qmd.releases == 1);
            REQUIRE(view->Read<u32>(va + 0x900) == 99);
            REQUIRE_FALSE(accepted_release.HasPendingQmd());
            qmd[12] = 1;
            REQUIRE(write_qmd());
            view->Write<u32>(va + 0x900, 77);
            REQUIRE(accepted_release.SignalQmd(va + 0x800, 3, release_host.Bind()).error ==
                    Qmd::Error::DispatchFailed);
            REQUIRE(view->Read<u32>(va + 0x900) == 77);
            REQUIRE(release_host.releases == 1);
            REQUIRE(accepted_release.HasQmdFailure());
        }
        selected = 0xfe;
        REQUIRE(pool.Map(&selected, spaces, backing, space, va + PageSize, va + 2 * PageSize - 1,
                         handle, 0, options)
                    .IsError());
        REQUIRE(selected == 0xfe);
        REQUIRE(view->Read<u32>(va) == 0x12345678);
        REQUIRE(view->MapSparse(va + size, PageSize, false) == va + size);
        REQUIRE(pool.Map(&selected, spaces, backing, space, va + size, va + size + PageSize - 1,
                         handle, 0, options)
                    .IsError());
        view->Unmap(va + size, PageSize);
        REQUIRE(pool.Unmap(space, va, va + PageSize - 1).IsSuccess());
        REQUIRE_FALSE(view->GpuToCpuAddress(va));
        REQUIRE(view->Read<u32>(va + PageSize) == 0xabcdef01);
        REQUIRE(pool.Map(&selected, spaces, backing, space, va, va + PageSize - 1, handle, PageSize,
                         options)
                    .IsSuccess());
        constexpr auto split_va = va + 8 * PageSize;
        REQUIRE(pool.Map(&selected, spaces, backing, space, split_va, split_va + 4 * PageSize - 1,
                         handle, 0, options)
                    .IsSuccess());
        REQUIRE(pool.Unmap(space, split_va + PageSize, split_va + 3 * PageSize - 1).IsSuccess());
        REQUIRE(view->GpuToCpuAddress(split_va).has_value());
        REQUIRE_FALSE(view->GpuToCpuAddress(split_va + PageSize));
        REQUIRE_FALSE(view->GpuToCpuAddress(split_va + 2 * PageSize));
        REQUIRE(view->GpuToCpuAddress(split_va + 3 * PageSize).has_value());
        REQUIRE(pool.Unmap(space, split_va, split_va + 4 * PageSize - 1) == Result{0x280f3});
        REQUIRE(view->GpuToCpuAddress(split_va).has_value());
        REQUIRE(pool.Unmap(space, split_va, split_va + PageSize - 1).IsSuccess());
        REQUIRE(
            pool.Unmap(space, split_va + 3 * PageSize, split_va + 4 * PageSize - 1).IsSuccess());
        auto* event = KEvent::Create(kernel);
        REQUIRE(event != nullptr);
        event->Initialize(kernel, nullptr);
        KEvent::Register(kernel, event);
        SCOPE_EXIT {
            event->GetReadableEvent().Close(kernel);
            event->Close(kernel);
        };
        REQUIRE(pool.RetainReference(handle, GetInteger(address) + PageSize, PageSize, PageSize)
                    .IsSuccess());
        REQUIRE(pool.Unregister(handle, event).IsSuccess());
        REQUIRE_FALSE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(devices->Destroy(device) == GraphicsDeviceTable::InvalidDevice);
        REQUIRE(view->Read<u32>(va) == 0x12345678);
        REQUIRE(pool.Unmap(space, va, va + size - 1).IsSuccess());
        REQUIRE_FALSE(view->GpuToCpuAddress(va));
        REQUIRE_FALSE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(alias_view->Read<u32>(va) == 0x12345678);
        REQUIRE(devices->Destroy(device) == GraphicsDeviceTable::InvalidDevice);
        REQUIRE(pool.Unmap(alias_space, va, va + size - 1).IsSuccess());
        REQUIRE_FALSE(alias_view->GpuToCpuAddress(va));
        REQUIRE_FALSE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(devices->Destroy(device) == GraphicsDeviceTable::InvalidDevice);
        REQUIRE(pool.ReleaseReference(handle).IsSuccess());
        REQUIRE(event->GetReadableEvent().IsSignaled(kernel));
        REQUIRE(devices->Destroy(device).IsSuccess());
        REQUIRE(pool.Unregister(handle) == Result{0x280f3});
    }
    SECTION("sparse GPU regions own virtual holes and resident overlays independently") {
        using namespace Service::GSV;
        Tegra::GPU gpu{system, false, false};
        Null::RasterizerNull rasterizer{gpu};
        Tegra::MaxwellDeviceMemoryManager backing{system.DeviceMemory()};
        backing.BindInterface(&rasterizer);
        GpuAddressSpacePool spaces;
        auto manager =
            std::make_unique<Tegra::MemoryManager>(system, backing, 40, 1ULL << 40, 16, 12);
        manager->BindRasterizer(&rasterizer);
        u32 space{};
        REQUIRE(spaces.Add(&space, std::move(manager), 0).IsSuccess());
        const auto view = spaces.Acquire(space);
        auto devices = std::make_shared<GraphicsDeviceTable>();
        u8 device{}, other_device{};
        REQUIRE(devices->Create(&device, 1, 0, 0).IsSuccess());
        REQUIRE(devices->Create(&other_device, 2, 0, 0).IsSuccess());
        constexpr u64 va = 0x8000000000;
        constexpr u64 size = 250ULL << 20;
        constexpr u64 last = va + size - 1;
        const auto ram_before = kernel.MemoryManager().GetFreeSize();
        {
            GpuMemoryPool pool{kernel, &process, devices};
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last, 11, 1, device).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last, 12, 2, device).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last, 12, 1, 8).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space + 8, va, last, 12, 1, device).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va + 1, last, 12, 1, device).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last - 1, 12, 1, device).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, va - 1, 12, 1, device).IsError());
            REQUIRE(
                pool.CreateSparseRegion(spaces, space, 0, PageSize - 1, 12, 1, device).IsError());
            REQUIRE(
                pool.CreateSparseRegion(spaces, space, va, 1ULL << 40, 12, 1, device).IsError());
            REQUIRE(devices->Get(device)->memory_references == 0);
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last, 12, 1, device).IsSuccess());
            REQUIRE(kernel.MemoryManager().GetFreeSize() == ram_before);
            REQUIRE(devices->Destroy(device).IsError());
            REQUIRE(devices->Get(device)->memory_references == 1);
            REQUIRE_FALSE(view->GpuToCpuAddress(va));
            REQUIRE_FALSE(view->GpuToCpuAddress(last));
            REQUIRE(view->GetPointer(va) == nullptr);
            REQUIRE(view->GetSubmappedRange(va, size).empty());
            REQUIRE(view->Read<u32>(va) == 0);
            view->Write<u32>(va, 0xffffffff);
            REQUIRE(view->Read<u32>(va) == 0);
            REQUIRE(view->Read<u64>(last - sizeof(u64) + 1) == 0);
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last, 12, 1, device).IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va + PageSize, last, 12, 1, device)
                        .IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va - PageSize, va + PageSize - 1, 12, 1,
                                            device)
                        .IsError());
            GpuMemoryPool foreign{kernel, &process, devices};
            REQUIRE(foreign.CreateSparseRegion(spaces, space, va, last, 12, 1, device).IsError());
            REQUIRE(devices->Get(device)->memory_references == 1);

            REQUIRE(table
                        .SetGpuMemoryAttribute(address, 4 * PageSize, KMemoryAttribute::GpuSharable,
                                               KMemoryAttribute::GpuSharable)
                        .IsSuccess());
            u32 handle{}, foreign_handle{};
            REQUIRE(foreign
                        .Register(&foreign_handle, GetInteger(address),
                                  GetInteger(address) + 4 * PageSize - 1, 0, device)
                        .IsSuccess());
            const GpuMemoryPool::MapOptions options{0, 0, 6, 6, 0, device, 1};
            u8 kind{0xff};
            REQUIRE(foreign
                        .Map(&kind, spaces, backing, space, va, va + PageSize - 1, foreign_handle,
                             0, options)
                        .IsError());
            REQUIRE(kind == 0xff);
            REQUIRE(foreign.Unregister(foreign_handle).IsSuccess());
            REQUIRE(pool.Register(&handle, GetInteger(address),
                                  GetInteger(address) + 4 * PageSize - 1, 0, device)
                        .IsSuccess());
            REQUIRE(
                pool.Map(&kind, spaces, backing, space, va, va + PageSize - 1, handle, 0, options)
                    .IsError());
            REQUIRE(kind == 0xff);
            REQUIRE(pool.CreateSparseRegion(spaces, space, va + size, va + size + PageSize - 1, 12,
                                            1, other_device)
                        .IsSuccess());
            REQUIRE(pool.MapSparseMemory(&kind, spaces, backing, space, va + size,
                                         va + size + PageSize - 1, handle, 0, options)
                        .IsError());
            process.GetMemory().Write32(address, 0x12345678);
            REQUIRE(pool.MapSparseMemory(&kind, spaces, backing, space, va + PageSize,
                                         va + 3 * PageSize - 1, handle, 0, options)
                        .IsSuccess());
            REQUIRE(kind == 6);
            REQUIRE(view->Read<u32>(va + PageSize) == 0x12345678);
            REQUIRE(view->Read<u32>(va + PageSize - 2) == 0x56780000);
            view->Write<u32>(va + PageSize - 2, 0xabcdef01);
            REQUIRE(process.GetMemory().Read32(address) == 0x1234abcd);
            REQUIRE(view->Read<u16>(va + PageSize - 2) == 0);
            REQUIRE(view->Read<u32>(va + 3 * PageSize) == 0);
            REQUIRE(view->GetSubmappedRange(va, 4 * PageSize).size() == 1);
            REQUIRE(
                pool.Map(&kind, spaces, backing, space, va - PageSize, va - 1, handle, 0, options)
                    .IsSuccess());
            u64 ordinary_unmapped{};
            REQUIRE(pool.Unmap(space, va - PageSize, va + size + PageSize - 1, &ordinary_unmapped)
                        .IsSuccess());
            REQUIRE(ordinary_unmapped == PageSize);
            REQUIRE_FALSE(view->GpuToCpuAddress(va - PageSize));
            REQUIRE(view->Read<u32>(va + PageSize) == 0x1234abcd);
            REQUIRE_FALSE(view->MapIfFree(va, 0, PageSize, Tegra::PTEKind::PITCH));
            REQUIRE_FALSE(view->ReserveSparseIfFree(va + PageSize, PageSize));
            REQUIRE(pool.DestroySparseRegion(space, va, last - PageSize).IsError());
            REQUIRE(pool.DestroySparseRegion(space, va + PageSize, last).IsError());
            REQUIRE(view->Read<u32>(va + PageSize) == 0x1234abcd);

            u64 unmapped{~u64{0}};
            REQUIRE(pool.Unmap(space, va, va + 4 * PageSize - 1, &unmapped).IsSuccess());
            REQUIRE(unmapped == 0);
            REQUIRE(view->Read<u32>(va + PageSize) == 0x1234abcd);
            REQUIRE(
                pool.UnmapSparseMemory(space, va, va + 4 * PageSize - 1, &unmapped).IsSuccess());
            REQUIRE(unmapped == 2 * PageSize);
            REQUIRE(
                pool.UnmapSparseMemory(space, va, va + 4 * PageSize - 1, &unmapped).IsSuccess());
            REQUIRE(unmapped == 0);
            REQUIRE(view->GetSubmappedRange(va, size).empty());
            REQUIRE_FALSE(view->ReserveSparseIfFree(va, PageSize));
            REQUIRE(view->Read<u32>(va + PageSize) == 0);
            REQUIRE(pool.MapSparseMemory(&kind, spaces, backing, space, va, va + 4 * PageSize - 1,
                                         handle, 0, options)
                        .IsSuccess());
            REQUIRE(pool.UnmapSparseMemory(space, va + PageSize, va + 3 * PageSize - 1, &unmapped)
                        .IsSuccess());
            REQUIRE(unmapped == 2 * PageSize);
            REQUIRE(view->GpuToCpuAddress(va).has_value());
            REQUIRE_FALSE(view->GpuToCpuAddress(va + PageSize));
            REQUIRE(view->GpuToCpuAddress(va + 3 * PageSize).has_value());
            REQUIRE(pool.MapSparseMemory(&kind, spaces, backing, space, va + PageSize,
                                         va + 2 * PageSize - 1, handle, PageSize, options)
                        .IsSuccess());
            REQUIRE(pool.Unregister(handle).IsSuccess());
            REQUIRE(pool.DestroySparseRegion(space, va, last).IsSuccess());
            REQUIRE_FALSE(view->GpuToCpuAddress(va));
            REQUIRE_FALSE(view->GpuToCpuAddress(va + 3 * PageSize));
            REQUIRE(pool.Unregister(handle).IsError());
            REQUIRE(pool.DestroySparseRegion(space, va, last).IsError());
            REQUIRE(devices->Get(device)->memory_references == 0);
            REQUIRE(view->ReserveSparseIfFree(va, size));
            view->Unmap(va, size);

            constexpr u64 large_size = 1ULL << 16;
            REQUIRE(pool.CreateSparseRegion(spaces, space, va + PageSize, va + large_size - 1, 16,
                                            0, device)
                        .IsError());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, va + large_size - 1, 16, 0, device)
                        .IsSuccess());
            REQUIRE(view->Read<u64>(va) == 0);
            REQUIRE(pool.UnmapSparseMemory(space, va, va + PageSize - 1).IsError());
            REQUIRE(pool.UnmapSparseMemory(space, va, va + large_size - 1, &unmapped).IsSuccess());
            REQUIRE(unmapped == 0);
            REQUIRE(pool.DestroySparseRegion(space, va, va + large_size - 1).IsSuccess());
            REQUIRE(pool.CreateSparseRegion(spaces, space, va, last, 12, 0, device).IsSuccess());
        }
        REQUIRE(kernel.MemoryManager().GetFreeSize() == ram_before);
        REQUIRE(view->GetSubmappedRange(va, size).empty());
        REQUIRE(view->ReserveSparseIfFree(va, size));
        view->Unmap(va, size);
        REQUIRE(devices->Destroy(device).IsSuccess());
        REQUIRE(devices->Destroy(other_device).IsSuccess());
    }
    SECTION("sparse batches validate the whole packet and preserve native kind aliases") {
        using namespace Service::GSV;
        Tegra::GPU gpu{system, false, false};
        Null::RasterizerNull rasterizer{gpu};
        Tegra::MaxwellDeviceMemoryManager backing{system.DeviceMemory()};
        backing.BindInterface(&rasterizer);
        GpuAddressSpacePool spaces;
        auto manager =
            std::make_unique<Tegra::MemoryManager>(system, backing, 40, 1ULL << 40, 16, 12);
        manager->BindRasterizer(&rasterizer);
        u32 space{};
        REQUIRE(spaces.Add(&space, std::move(manager), 0).IsSuccess());
        const auto view = spaces.Acquire(space);
        auto devices = std::make_shared<GraphicsDeviceTable>();
        u8 device{};
        REQUIRE(devices->Create(&device, 1, 0, 0).IsSuccess());
        GpuMemoryPool pool{kernel, &process, devices};
        constexpr u64 va = 0x8800000000;
        constexpr u64 region_size = 16 * PageSize;
        REQUIRE(pool.CreateSparseRegion(spaces, space, va, va + region_size - 1, 12, 1, device)
                    .IsSuccess());
        REQUIRE(table
                    .SetGpuMemoryAttribute(address, 4 * PageSize, KMemoryAttribute::GpuSharable,
                                           KMemoryAttribute::GpuSharable)
                    .IsSuccess());
        u32 handle{};
        REQUIRE(pool.Register(&handle, GetInteger(address), GetInteger(address) + 4 * PageSize - 1,
                              0, device)
                    .IsSuccess());
        process.GetMemory().Write32(address, 0xabcdef01);
        process.GetMemory().Write32(address + PageSize, 0x12345678);
        const auto request = [](u64 gpu_address, u64 size, u32 memory_handle, u64 offset = 0,
                                u8 kind = 6, u8 fallback = 6) {
            std::array<u8, 40> bytes{};
            const auto put = [&](size_t index, const auto& value) {
                std::memcpy(bytes.data() + index, &value, sizeof(value));
            };
            put(0, u64_le{gpu_address});
            put(8, u64_le{size});
            put(24, u32_le{memory_handle});
            put(32, u64_le{offset});
            if (memory_handle != 0) {
                bytes[17] = 1;
                bytes[19] = kind;
                bytes[20] = fallback;
            }
            return bytes;
        };
        u64 processed{~u64{0}};
        bool accepted{};
        const auto execute = [&](std::span<const u8> input, std::span<u8> kinds, u64 version = 1,
                                 u64 stride = 40, u8 sync = 1) {
            return pool.ExecuteSparseMapRequests(&processed, kinds, spaces, backing, space, input,
                                                 version, stride, sync, &accepted);
        };
        std::vector<u8> batch;
        constexpr std::array<u8, 6> requested{6, 8, 11, 12, 9, 13};
        constexpr std::array<u8, 6> fallback{6, 6, 1, 3, 6, 4};
        constexpr std::array layouts{Tegra::PTEKind::GENERIC_16BX2, Tegra::PTEKind::GENERIC_16BX2,
                                     Tegra::PTEKind::Z16,           Tegra::PTEKind::S8Z24,
                                     Tegra::PTEKind::GENERIC_16BX2, Tegra::PTEKind::ZF32_X24S8};
        for (size_t i = 0; i < requested.size(); ++i) {
            const auto bytes =
                request(va + i * PageSize, PageSize, handle, 0, requested[i], fallback[i]);
            batch.insert(batch.end(), bytes.begin(), bytes.end());
        }
        std::array<u8, 8> kinds;
        kinds.fill(0xcc);
        REQUIRE(execute(batch, kinds).IsSuccess());
        REQUIRE(accepted);
        REQUIRE(processed == 6);
        for (size_t i = 0; i < requested.size(); ++i) {
            REQUIRE(kinds[i] == fallback[i]);
            REQUIRE(view->GetPageKind(va + i * PageSize) == layouts[i]);
            REQUIRE(view->GpuToCpuAddress(va + i * PageSize) == view->GpuToCpuAddress(va));
            REQUIRE(view->Read<u32>(va + i * PageSize) == 0xabcdef01);
        }
        REQUIRE(kinds[6] == 0xcc);
        REQUIRE(kinds[7] == 0xcc);
        view->Write<u32>(va + 3 * PageSize, 0xfedcba98);
        REQUIRE(process.GetMemory().Read32(address) == 0xfedcba98);
        REQUIRE(view->Read<u32>(va + 5 * PageSize) == 0xfedcba98);
        REQUIRE(process.GetMemory().Read32(address + PageSize) == 0x12345678);
        REQUIRE(execute(batch, std::span{kinds}.first(2)).IsSuccess());
        REQUIRE(processed == 6);
        auto aliased = batch;
        REQUIRE(execute(aliased, std::span{aliased}.first(6)).IsSuccess());
        REQUIRE(processed == 6);
        REQUIRE(std::equal(aliased.begin(), aliased.begin() + 6, fallback.begin()));

        const auto unmap = request(va, PageSize, 0);
        auto invalid = request(va + PageSize, PageSize, handle);
        const auto rejected = [&](std::span<const u8> input, u64 version = 1, u64 stride = 40,
                                  u8 sync = 1) {
            processed = ~u64{0};
            accepted = true;
            kinds.fill(0xcc);
            REQUIRE(execute(input, kinds, version, stride, sync).IsError());
            REQUIRE_FALSE(accepted);
            REQUIRE(processed == ~u64{0});
            REQUIRE(kinds[0] == 0xcc);
            REQUIRE(view->Read<u32>(va) == 0xfedcba98);
            REQUIRE(view->Read<u32>(va + PageSize) == 0xfedcba98);
        };
        kinds.fill(0xcc);
        REQUIRE(execute({}, kinds).IsSuccess());
        REQUIRE(accepted);
        REQUIRE(processed == 0);
        REQUIRE(kinds[0] == 0xcc);
        rejected(std::span{invalid}.first(39));
        rejected(invalid, 0);
        rejected(invalid, 1, 0);
        rejected(invalid, 1, 41);
        rejected(invalid, 1, 40, 2);
        for (const auto index : {12U, 16U, 17U, 18U, 21U, 22U, 24U, 28U, 36U}) {
            auto malformed = invalid;
            malformed[index] = 0xff;
            std::vector<u8> two{unmap.begin(), unmap.end()};
            two.insert(two.end(), malformed.begin(), malformed.end());
            rejected(two);
        }
        auto bad_unmap = unmap;
        bad_unmap[17] = 1;
        rejected(bad_unmap);
        auto bad_kind = invalid;
        bad_kind[19] = 7;
        rejected(bad_kind);
        bad_kind[19] = 8;
        bad_kind[20] = 8;
        rejected(bad_kind);
        rejected(request(va + region_size, PageSize, handle));
        rejected(request(va, PageSize, handle, 4 * PageSize));
        rejected(request(va, PageSize + 1, handle));
        rejected(request(va + 1, PageSize, handle));
        rejected(request((1ULL << 40) - PageSize, 2 * PageSize, handle));
        rejected(request(va, PageSize, handle, 1ULL << 37));

        const auto retained = *view->GpuToCpuAddress(va + PageSize);
        REQUIRE(view->Map(va + 12 * PageSize, retained, PageSize, Tegra::PTEKind::GENERIC_16BX2,
                          false) == va + 12 * PageSize);
        const auto first = request(va + 10 * PageSize, PageSize, handle);
        const auto collision = request(va + 12 * PageSize, PageSize, handle);
        std::vector<u8> partial{first.begin(), first.end()};
        partial.insert(partial.end(), collision.begin(), collision.end());
        kinds.fill(0xcc);
        REQUIRE(execute(partial, kinds).IsError());
        REQUIRE(accepted);
        REQUIRE(processed == 1);
        REQUIRE(kinds[0] == 6);
        REQUIRE(kinds[1] == 0xcc);
        REQUIRE(view->Read<u32>(va + 10 * PageSize) == 0xfedcba98);
        REQUIRE(view->GpuToCpuAddress(va + 12 * PageSize) == retained);
        view->Unmap(va + 12 * PageSize, PageSize, true);

        for (u8 kind = 0; kind <= 6; ++kind) {
            const auto bytes = request(va + 8 * PageSize, PageSize, handle, PageSize, kind, 7);
            REQUIRE(execute(bytes, {}).IsSuccess());
            REQUIRE(processed == 1);
            REQUIRE(view->Read<u32>(va + 8 * PageSize) == 0x12345678);
        }
        const auto remap = request(va, PageSize, handle, PageSize);
        REQUIRE(execute(remap, kinds).IsSuccess());
        REQUIRE(view->Read<u32>(va) == 0x12345678);
        REQUIRE(view->Read<u32>(va + PageSize) == 0xfedcba98);
        REQUIRE(execute(unmap, kinds).IsSuccess());
        REQUIRE(processed == 1);
        REQUIRE(kinds[0] == 0);
        REQUIRE_FALSE(view->GpuToCpuAddress(va));
        REQUIRE(view->Read<u32>(va) == 0);
        REQUIRE(view->Read<u32>(va + PageSize) == 0xfedcba98);
        REQUIRE(execute(unmap, kinds).IsSuccess());
        REQUIRE(processed == 1);
        REQUIRE(pool.Unregister(handle).IsSuccess());
        REQUIRE(pool.DestroySparseRegion(space, va, va + region_size - 1).IsSuccess());
        REQUIRE(pool.Unregister(handle).IsError());
        REQUIRE(view->GetSubmappedRange(va, region_size).empty());
        REQUIRE(devices->Destroy(device).IsSuccess());
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
