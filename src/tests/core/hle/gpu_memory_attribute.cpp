// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "core/hle/kernel/k_memory_block_manager.h"

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
