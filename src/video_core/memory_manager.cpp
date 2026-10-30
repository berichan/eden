// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "memory_manager.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_shim.h"
#include "video_core/invalidation_accumulator.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "video_core/texture_cache/util.h"

namespace Tegra {
using Tegra::Memory::GuestMemoryFlags;

std::atomic<size_t> MemoryManager::unique_identifier_generator{};

MemoryManager::MemoryManager(Core::System& system_, MaxwellDeviceMemoryManager& memory_, u64 address_space_bits_, GPUVAddr split_address_, u64 big_page_bits_, u64 page_bits_)
    : system{system_}, memory{memory_}, address_space_bits{address_space_bits_}
    , split_address{split_address_}, page_bits{page_bits_}, big_page_bits{big_page_bits_}
    , entries{}, big_entries{}
    , page_table{address_space_bits, address_space_bits + page_bits - 38, page_bits != big_page_bits ? page_bits : 0}
    , kind_map{PTEKind::INVALID}, unique_identifier{unique_identifier_generator.fetch_add(1, std::memory_order_acq_rel)}
    , accumulator{}
{
    address_space_size = 1ULL << address_space_bits;
    page_size = 1ULL << page_bits;
    page_mask = page_size - 1ULL;
    big_page_size = 1ULL << big_page_bits;
    big_page_mask = big_page_size - 1ULL;
    const u64 page_table_bits = address_space_bits - page_bits;
    const u64 big_page_table_bits = address_space_bits - big_page_bits;
    const u64 page_table_size = 1ULL << page_table_bits;
    const u64 big_page_table_size = 1ULL << big_page_table_bits;
    page_table_mask = page_table_size - 1;
    big_page_table_mask = big_page_table_size - 1;

    big_page_table_dev.ResizeAndClear(big_page_table_size);
    big_entries.resize(big_page_table_size / 32, 0);
    big_page_continuous.resize(big_page_table_size / continuous_bits, 0);
    entries.resize(page_table_size / 32, 0);
}

MemoryManager::MemoryManager(Core::System& system_, u64 address_space_bits_, GPUVAddr split_address_, u64 big_page_bits_, u64 page_bits_)
    : MemoryManager(system_, system_.Host1x().MemoryManager(), address_space_bits_, split_address_, big_page_bits_, page_bits_)
{}

MemoryManager::~MemoryManager() = default;

MemoryManager::EntryType MemoryManager::GetEntry(size_t position, bool is_big_page) const {
    if (is_big_page) {
        position = position >> big_page_bits;
        const u64 entry_mask = big_entries[position / 32];
        const size_t sub_index = position % 32;
        return static_cast<EntryType>((entry_mask >> (2 * sub_index)) & 0x03ULL);
    } else {
        position = position >> page_bits;
        const u64 entry_mask = entries[position / 32];
        const size_t sub_index = position % 32;
        return static_cast<EntryType>((entry_mask >> (2 * sub_index)) & 0x03ULL);
    }
}

void MemoryManager::SetEntry(size_t position, MemoryManager::EntryType entry, bool is_big_page) {
    mapping_generation.fetch_add(1, std::memory_order_release);
    if (is_big_page) {
        position = position >> big_page_bits;
        const u64 entry_mask = big_entries[position / 32];
        const size_t sub_index = position % 32;
        big_entries[position / 32] =
            (~(3ULL << sub_index * 2) & entry_mask) | (static_cast<u64>(entry) << sub_index * 2);
    } else {
        position = position >> page_bits;
        const u64 entry_mask = entries[position / 32];
        const size_t sub_index = position % 32;
        entries[position / 32] =
            (~(3ULL << sub_index * 2) & entry_mask) | (static_cast<u64>(entry) << sub_index * 2);
    }
}

PTEKind MemoryManager::GetPageKind(GPUVAddr gpu_addr) const {
    std::unique_lock<std::mutex> lock(guard);
    return kind_map.GetValueAt(gpu_addr);
}

inline bool MemoryManager::IsBigPageContinuous(size_t big_page_index) const {
    const u64 entry_mask = big_page_continuous[big_page_index / continuous_bits];
    const size_t sub_index = big_page_index % continuous_bits;
    return ((entry_mask >> sub_index) & 0x1ULL) != 0;
}

inline void MemoryManager::SetBigPageContinuous(size_t big_page_index, bool value) {
    const u64 continuous_mask = big_page_continuous[big_page_index / continuous_bits];
    const size_t sub_index = big_page_index % continuous_bits;
    big_page_continuous[big_page_index / continuous_bits] =
        (~(1ULL << sub_index) & continuous_mask) | (value ? 1ULL << sub_index : 0);
}

GPUVAddr MemoryManager::PageTableOp(GPUVAddr gpu_addr, [[maybe_unused]] DAddr dev_addr, size_t size, PTEKind kind, MemoryManager::EntryType entry_type) {
    [[maybe_unused]] u64 remaining_size{size};
    if (entry_type == EntryType::Mapped) {
        page_table.ReserveRange(gpu_addr, size);
    }
    for (u64 offset{}; offset < size; offset += page_size) {
        const GPUVAddr current_gpu_addr = gpu_addr + offset;
        [[maybe_unused]] const auto current_entry_type = GetEntry(current_gpu_addr, false);
        SetEntry(current_gpu_addr, entry_type, false);
        if (current_entry_type != entry_type) {
            rasterizer->ModifyGPUMemory(unique_identifier, current_gpu_addr, page_size);
        }
        if (entry_type == EntryType::Mapped) {
            const DAddr current_dev_addr = dev_addr + offset;
            const auto index = PageEntryIndex(current_gpu_addr, false);
            const u32 sub_value = static_cast<u32>(current_dev_addr >> cpu_page_bits);
            page_table[index] = sub_value;
        }
        remaining_size -= page_size;
    }
    kind_map.Map(gpu_addr, gpu_addr + size, kind);
    return gpu_addr;
}

GPUVAddr MemoryManager::BigPageTableOp(GPUVAddr gpu_addr, [[maybe_unused]] DAddr dev_addr, size_t size, PTEKind kind, MemoryManager::EntryType entry_type) {
    [[maybe_unused]] u64 remaining_size{size};
    for (u64 offset{}; offset < size; offset += big_page_size) {
        const GPUVAddr current_gpu_addr = gpu_addr + offset;
        [[maybe_unused]] const auto current_entry_type = GetEntry(current_gpu_addr, true);
        SetEntry(current_gpu_addr, entry_type, true);
        if (current_entry_type != entry_type) {
            rasterizer->ModifyGPUMemory(unique_identifier, current_gpu_addr, big_page_size);
        }
        if (entry_type == EntryType::Mapped) {
            const DAddr current_dev_addr = dev_addr + offset;
            const auto index = PageEntryIndex(current_gpu_addr, true);
            const u32 sub_value = static_cast<u32>(current_dev_addr >> cpu_page_bits);
            big_page_table_dev.Set(index, sub_value);
            const bool is_continuous = ([&] {
                uintptr_t base_ptr{
                    reinterpret_cast<uintptr_t>(memory.GetPointer<u8>(current_dev_addr))};
                if (base_ptr == 0) {
                    return false;
                }
                for (DAddr start_cpu = current_dev_addr + page_size;
                     start_cpu < current_dev_addr + big_page_size; start_cpu += page_size) {
                    base_ptr += page_size;
                    auto next_ptr = reinterpret_cast<uintptr_t>(memory.GetPointer<u8>(start_cpu));
                    if (next_ptr == 0 || base_ptr != next_ptr) {
                        return false;
                    }
                }
                return true;
            })();
            SetBigPageContinuous(index, is_continuous);
        }
        remaining_size -= big_page_size;
    }
    {
        std::unique_lock<std::mutex> lock(guard);
        kind_map.Map(gpu_addr, gpu_addr + size, kind);
    }
    return gpu_addr;
}

void MemoryManager::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

GPUVAddr MemoryManager::Map(GPUVAddr gpu_addr, DAddr dev_addr, std::size_t size, PTEKind kind, bool is_big_pages) {
    std::scoped_lock lock{mapping_guard};
    if (OverlapsSyncpointShim(gpu_addr, size)) {
        LOG_ERROR(HW_GPU, "Cannot overwrite read-only syncpoint mapping at {:#x}", gpu_addr);
        return 0;
    }
    if (is_big_pages)
        return BigPageTableOp(gpu_addr, dev_addr, size, kind, EntryType::Mapped);
    return PageTableOp(gpu_addr, dev_addr, size, kind, EntryType::Mapped);
}

bool MemoryManager::MapIfFree(GPUVAddr gpu_addr, DAddr dev_addr, std::size_t size, PTEKind kind) {
    return MapChecked(gpu_addr, dev_addr, size, kind, false);
}

bool MemoryManager::MapIntoSparse(GPUVAddr gpu_addr, DAddr dev_addr, size_t size, PTEKind kind) {
    return MapChecked(gpu_addr, dev_addr, size, kind, true);
}

bool MemoryManager::IsPageRange(GPUVAddr address, size_t size) const {
    return size != 0 && Common::IsAligned(address, page_size) &&
           Common::IsAligned(size, page_size) && address < address_space_size &&
           size <= address_space_size - address && !OverlapsSyncpointShim(address, size);
}

bool MemoryManager::MapChecked(GPUVAddr gpu_addr, DAddr dev_addr, size_t size, PTEKind kind,
                               bool sparse) {
    std::scoped_lock lock{mapping_guard};
    if (!IsPageRange(gpu_addr, size) || !Common::IsAligned(dev_addr, page_size) ||
        dev_addr >= (1ULL << MaxwellDeviceMemoryManager::AS_BITS) ||
        size > (1ULL << MaxwellDeviceMemoryManager::AS_BITS) - dev_addr) {
        return false;
    }
    for (auto address = gpu_addr; address < gpu_addr + size; address += page_size) {
        if (GetEntry(address, true) != EntryType::Free ||
            GetEntry(address, false) != (sparse ? EntryType::Reserved : EntryType::Free)) {
            return false;
        }
    }
    return PageTableOp(gpu_addr, dev_addr, size, kind, EntryType::Mapped) == gpu_addr;
}

bool MemoryManager::ReserveSparseIfFree(GPUVAddr gpu_addr, size_t size) {
    std::scoped_lock lock{mapping_guard};
    if (!IsPageRange(gpu_addr, size)) {
        return false;
    }
    for (auto address = gpu_addr; address < gpu_addr + size; address += page_size) {
        if (GetEntry(address, true) != EntryType::Free ||
            GetEntry(address, false) != EntryType::Free) {
            return false;
        }
    }
    // Use small HLE entries for either native page size so resident overlays never sit
    // underneath a large reserved entry that would hide their translations.
    return PageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Reserved) == gpu_addr;
}

GPUVAddr MemoryManager::MapSparse(GPUVAddr gpu_addr, std::size_t size, bool is_big_pages) {
    std::scoped_lock lock{mapping_guard};
    if (OverlapsSyncpointShim(gpu_addr, size)) {
        LOG_ERROR(HW_GPU, "Cannot overwrite read-only syncpoint mapping at {:#x}", gpu_addr);
        return 0;
    }
    if (is_big_pages)
        return BigPageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Reserved);
    return PageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Reserved);
}

void MemoryManager::Unmap(GPUVAddr gpu_addr, std::size_t size, bool preserve_sparse) {
    std::scoped_lock lock{mapping_guard};
    if (size == 0) {
        return;
    }
    if (OverlapsSyncpointShim(gpu_addr, size)) {
        LOG_ERROR(HW_GPU, "Cannot unmap driver-owned syncpoint mapping at {:#x}", gpu_addr);
        return;
    }
    GetSubmappedRangeImpl<false>(gpu_addr, size, page_stash);

    for (const auto& [map_addr, map_size] : page_stash) {
        rasterizer->UnmapMemory(map_addr, map_size);
    }
    page_stash.clear();

    BigPageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Free);
    PageTableOp(gpu_addr, 0, size, PTEKind::INVALID,
                preserve_sparse ? EntryType::Reserved : EntryType::Free);
}

bool MemoryManager::MapSyncpointShim(GPUVAddr gpu_addr, Host1x::SyncpointManager& syncpoints) {
    std::scoped_lock lock{mapping_guard};
    if (syncpoint_shim || !Common::IsAligned(gpu_addr, Host1x::SyncpointShim::Stride) ||
        gpu_addr >= address_space_size || Host1x::SyncpointShim::Size > address_space_size - gpu_addr) {
        return false;
    }
    for (GPUVAddr address = gpu_addr; address < gpu_addr + Host1x::SyncpointShim::Size;
         address += page_size) {
        if (GetEntry(address, true) != EntryType::Free ||
            GetEntry(address, false) != EntryType::Free) {
            return false;
        }
    }
    // Counter I/O has no RAM translation: reads sample completion atomics and cannot expose a
    // writable host pointer or a cached snapshot of the counters.
    syncpoint_shim = std::make_unique<Host1x::SyncpointShim>(syncpoints);
    syncpoint_shim_address = gpu_addr;
    syncpoint_shim_ready.store(true, std::memory_order_release);
    mapping_generation.fetch_add(1, std::memory_order_release);
    return true;
}

std::optional<GPUVAddr> MemoryManager::GetSyncpointShimAddress() const {
    return syncpoint_shim_ready.load(std::memory_order_acquire)
               ? std::optional{syncpoint_shim_address}
               : std::nullopt;
}

bool MemoryManager::OverlapsSyncpointShim(GPUVAddr address, size_t size) const {
    if (!syncpoint_shim_ready.load(std::memory_order_acquire) || size == 0) {
        return false;
    }
    return address < syncpoint_shim_address + Host1x::SyncpointShim::Size &&
           (address >= syncpoint_shim_address || size > syncpoint_shim_address - address);
}

bool MemoryManager::ReadSyncpointShim(GPUVAddr address, std::span<u8> output) const {
    return syncpoint_shim_ready.load(std::memory_order_acquire) &&
           address >= syncpoint_shim_address &&
           syncpoint_shim->Read(address - syncpoint_shim_address, output);
}

std::optional<DAddr> MemoryManager::GpuToCpuAddress(GPUVAddr gpu_addr) const {
    if (!IsWithinGPUAddressRange(gpu_addr)) [[unlikely]] {
        return std::nullopt;
    }
    if (GetEntry(gpu_addr, true) != EntryType::Mapped) [[unlikely]] {
        if (GetEntry(gpu_addr, false) != EntryType::Mapped)
            return std::nullopt;
        const DAddr dev_addr_base = DAddr(page_table[PageEntryIndex(gpu_addr, false)]) << cpu_page_bits;
        return dev_addr_base + (gpu_addr & page_mask);
    }
    const DAddr dev_addr_base = DAddr(big_page_table_dev[PageEntryIndex(gpu_addr, true)]) << cpu_page_bits;
    return dev_addr_base + (gpu_addr & big_page_mask);
}

std::optional<DAddr> MemoryManager::GpuToCpuAddress(GPUVAddr addr, std::size_t size) const {
    size_t page_index{addr >> page_bits};
    const size_t page_last{(addr + size + page_size - 1) >> page_bits};
    while (page_index < page_last) {
        const auto page_addr{GpuToCpuAddress(page_index << page_bits)};
        if (page_addr) {
            return page_addr;
        }
        ++page_index;
    }
    return std::nullopt;
}

template <typename T>
T MemoryManager::Read(GPUVAddr addr) const {
    if (auto page_pointer{GetPointer(addr)}; page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        T value;
        std::memcpy(&value, page_pointer, sizeof(T));
        return value;
    }

    T value{};
    if (ReadSyncpointShim(addr, std::span{reinterpret_cast<u8*>(&value), sizeof(value)})) {
        return value;
    }
    if (addr < address_space_size && sizeof(T) <= address_space_size - addr &&
        (GetEntry(addr, true) == EntryType::Reserved ||
         GetEntry(addr, false) == EntryType::Reserved)) {
        ReadBlockUnsafe(addr, &value, sizeof(value));
        return value;
    }
    ASSERT(false);

    return {};
}

template <typename T>
void MemoryManager::Write(GPUVAddr addr, T data) {
    if (auto page_pointer{GetPointer(addr)}; page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        std::memcpy(page_pointer, &data, sizeof(T));
        return;
    }

    if (OverlapsSyncpointShim(addr, sizeof(T))) {
        LOG_ERROR(HW_GPU, "Write to read-only syncpoint mapping at {:#x}", addr);
        return;
    }
    if (addr < address_space_size && sizeof(T) <= address_space_size - addr &&
        (GetEntry(addr, true) == EntryType::Reserved ||
         GetEntry(addr, false) == EntryType::Reserved)) {
        WriteBlockUnsafe(addr, &data, sizeof(data));
        return;
    }
    ASSERT(false);
}

template u8 MemoryManager::Read<u8>(GPUVAddr addr) const;
template u16 MemoryManager::Read<u16>(GPUVAddr addr) const;
template u32 MemoryManager::Read<u32>(GPUVAddr addr) const;
template u64 MemoryManager::Read<u64>(GPUVAddr addr) const;
template void MemoryManager::Write<u8>(GPUVAddr addr, u8 data);
template void MemoryManager::Write<u16>(GPUVAddr addr, u16 data);
template void MemoryManager::Write<u32>(GPUVAddr addr, u32 data);
template void MemoryManager::Write<u64>(GPUVAddr addr, u64 data);

u8* MemoryManager::GetPointer(GPUVAddr gpu_addr) {
    const auto address{GpuToCpuAddress(gpu_addr)};
    if (!address) {
        return {};
    }

    return memory.GetPointer<u8>(*address);
}

const u8* MemoryManager::GetPointer(GPUVAddr gpu_addr) const {
    const auto address{GpuToCpuAddress(gpu_addr)};
    if (!address) {
        return {};
    }

    return memory.GetPointer<u8>(*address);
}

#if defined(_MSC_VER) && !defined(__clang__) // no need for gcc / clang but msvc's compiler is more conservative with inlining.
#pragma inline_recursion(on)
#endif

template <typename FuncMapped, typename FuncReserved, typename FuncUnmapped>
inline void MemoryManager::MemoryOperation(GPUVAddr gpu_src_addr, std::size_t size, bool is_big_page, FuncMapped&& func_mapped, FuncReserved&& func_reserved, FuncUnmapped&& func_unmapped) const {
    using FuncMappedReturn =
        typename std::invoke_result<FuncMapped, std::size_t, std::size_t, std::size_t>::type;
    using FuncReservedReturn =
        typename std::invoke_result<FuncReserved, std::size_t, std::size_t, std::size_t>::type;
    using FuncUnmappedReturn =
        typename std::invoke_result<FuncUnmapped, std::size_t, std::size_t, std::size_t>::type;
    static constexpr bool BOOL_BREAK_MAPPED = std::is_same_v<FuncMappedReturn, bool>;
    static constexpr bool BOOL_BREAK_RESERVED = std::is_same_v<FuncReservedReturn, bool>;
    static constexpr bool BOOL_BREAK_UNMAPPED = std::is_same_v<FuncUnmappedReturn, bool>;
    u64 used_page_size;
    u64 used_page_mask;
    u64 used_page_bits;
    if (is_big_page) {
        used_page_size = big_page_size;
        used_page_mask = big_page_mask;
        used_page_bits = big_page_bits;
    } else {
        used_page_size = page_size;
        used_page_mask = page_mask;
        used_page_bits = page_bits;
    }
    std::size_t remaining_size{size};
    std::size_t page_index{gpu_src_addr >> used_page_bits};
    std::size_t page_offset{gpu_src_addr & used_page_mask};
    GPUVAddr current_address = gpu_src_addr;

    while (remaining_size > 0) {
        const std::size_t copy_amount{
            (std::min)(static_cast<std::size_t>(used_page_size) - page_offset, remaining_size)};
        auto entry = GetEntry(current_address, is_big_page);
        if (entry == EntryType::Mapped) [[likely]] {
            if constexpr (BOOL_BREAK_MAPPED) {
                if (func_mapped(page_index, page_offset, copy_amount)) {
                    return;
                }
            } else {
                func_mapped(page_index, page_offset, copy_amount);
            }

        } else if (entry == EntryType::Reserved) {
            if constexpr (BOOL_BREAK_RESERVED) {
                if (func_reserved(page_index, page_offset, copy_amount)) {
                    return;
                }
            } else {
                func_reserved(page_index, page_offset, copy_amount);
            }

        } else [[unlikely]] {
            if constexpr (BOOL_BREAK_UNMAPPED) {
                if (func_unmapped(page_index, page_offset, copy_amount)) {
                    return;
                }
            } else {
                func_unmapped(page_index, page_offset, copy_amount);
            }
        }
        page_index++;
        page_offset = 0;
        remaining_size -= copy_amount;
        current_address += copy_amount;
    }
}

void MemoryManager::ReadBlockImpl(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size, [[maybe_unused]] VideoCommon::CacheType which, bool unsafe) const {
    if (OverlapsSyncpointShim(gpu_src_addr, size)) {
        const auto end = syncpoint_shim_address + Host1x::SyncpointShim::Size;
        if (gpu_src_addr < syncpoint_shim_address) {
            const size_t prefix = syncpoint_shim_address - gpu_src_addr;
            ReadBlockImpl(gpu_src_addr, dest_buffer, prefix, which, unsafe);
            ReadBlockImpl(syncpoint_shim_address, static_cast<u8*>(dest_buffer) + prefix,
                          size - prefix, which, unsafe);
        } else {
            const size_t amount = (std::min)(size, static_cast<size_t>(end - gpu_src_addr));
            const bool read = ReadSyncpointShim(
                gpu_src_addr, std::span{static_cast<u8*>(dest_buffer), amount});
            if (!read) {
                LOG_ERROR(HW_GPU, "Unsupported syncpoint register read at {:#x}, size={:#x}",
                          gpu_src_addr, amount);
                std::memset(dest_buffer, 0, amount);
            }
            if (amount < size) {
                ReadBlockImpl(end, static_cast<u8*>(dest_buffer) + amount, size - amount, which,
                              unsafe);
            }
        }
        return;
    }
    auto set_to_zero = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset, std::size_t copy_amount) {
        std::memset(dest_buffer, 0, copy_amount);
        dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
    };
    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
        }
        u8* physical = memory.GetPointer<u8>(dev_addr_base);
        std::memcpy(dest_buffer, physical, copy_amount);
        dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
        }
        if (!IsBigPageContinuous(page_index)) [[unlikely]] {
            memory.ReadBlockUnsafe(dev_addr_base, dest_buffer, copy_amount);
        } else {
            u8* physical = memory.GetPointer<u8>(dev_addr_base);
            std::memcpy(dest_buffer, physical, copy_amount);
        }
        dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
    };
    auto read_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, set_to_zero, set_to_zero);
    };
    MemoryOperation(gpu_src_addr, size, true, mapped_big, set_to_zero, read_short_pages);
}

void MemoryManager::ReadBlock(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size, VideoCommon::CacheType which) const {
    ReadBlockImpl(gpu_src_addr, dest_buffer, size, which, false);
}

void MemoryManager::ReadBlockUnsafe(GPUVAddr gpu_src_addr, void* dest_buffer, const std::size_t size) const {
    ReadBlockImpl(gpu_src_addr, dest_buffer, size, VideoCommon::CacheType::None, true);
}

bool MemoryManager::IsValidAccessRange(GPUVAddr address, size_t size, bool write) const {
    if (address > address_space_size || size > address_space_size - address) {
        return false;
    }
    const u64 granule = (std::min)(page_size, u64{1} << cpu_page_bits);
    while (size != 0) {
        if (syncpoint_shim_ready.load(std::memory_order_acquire) &&
            address >= syncpoint_shim_address &&
            address < syncpoint_shim_address + Host1x::SyncpointShim::Size) {
            const auto amount = (std::min)(u64{size}, syncpoint_shim_address +
                                                          Host1x::SyncpointShim::Size - address);
            if (write ||
                !Host1x::SyncpointShim::IsValidRead(address - syncpoint_shim_address, amount)) {
                return false;
            }
            address += amount;
            size -= amount;
            continue;
        }
        const size_t amount = (std::min)(u64{size}, granule - (address & (granule - 1)));
        auto entry = GetEntry(address, true);
        if (entry == EntryType::Free) {
            entry = GetEntry(address, false);
        }
        if (entry == EntryType::Free) {
            return false;
        }
        if (entry == EntryType::Mapped) {
            const auto device = GpuToCpuAddress(address);
            if (!device || *device >= (u64{1} << MaxwellDeviceMemoryManager::AS_BITS) ||
                amount > (u64{1} << MaxwellDeviceMemoryManager::AS_BITS) - *device ||
                !memory.GetPointer<u8>(*device) || !memory.GetPointer<u8>(*device + amount - 1)) {
                return false;
            }
        }
        address += amount;
        size -= amount;
    }
    return true;
}

bool MemoryManager::ReadBlockChecked(GPUVAddr address, std::span<u8> output) const {
    std::scoped_lock lock{mapping_guard};
    if (!IsValidAccessRange(address, output.size(), false)) {
        return false;
    }
    ReadBlock(address, output.data(), output.size());
    return true;
}

bool MemoryManager::IsReadableRange(GPUVAddr address, size_t size) const {
    std::scoped_lock lock{mapping_guard};
    return IsValidAccessRange(address, size, false);
}

bool MemoryManager::IsWritableRange(GPUVAddr address, size_t size) const {
    std::scoped_lock lock{mapping_guard};
    return IsValidAccessRange(address, size, true);
}

bool MemoryManager::WriteBlockChecked(GPUVAddr address, std::span<const u8> input) {
    std::scoped_lock lock{mapping_guard};
    if (!IsValidAccessRange(address, input.size(), true)) {
        return false;
    }
    WriteBlock(address, input.data(), input.size());
    return true;
}

MemoryManager::MappingLease::MappingLease(MemoryManager& manager_)
    : manager{&manager_}, lock{manager_.mapping_guard} {}

MemoryManager::MappingLease MemoryManager::AcquireMappingLease() {
    return MappingLease{*this};
}

u64 MemoryManager::MappingGeneration() const {
    return mapping_generation.load(std::memory_order_acquire);
}

std::optional<DAddr> MemoryManager::MappingLease::ResolveContiguous(GPUVAddr address, size_t size,
                                                                    bool write) const {
    if (!lock.owns_lock() || size == 0 || !manager->IsValidAccessRange(address, size, write) ||
        !manager->IsContinuousRange(address, size)) {
        return std::nullopt;
    }
    return manager->GpuToCpuAddress(address);
}

size_t MemoryManager::MappingLease::ContiguousPrefix(GPUVAddr address, size_t max_size, bool write,
                                                     std::stop_token stop_token) const {
    if (!lock.owns_lock() || address >= manager->address_space_size) {
        return 0;
    }
    const size_t limit = (std::min)(u64{max_size}, manager->address_space_size - address);
    const u64 granule = (std::min)(manager->page_size, u64{1} << cpu_page_bits);
    std::optional<DAddr> base;
    size_t offset{};
    while (offset < limit) {
        if (stop_token.stop_requested()) {
            return 0;
        }
        const GPUVAddr current = address + offset;
        const size_t amount = (std::min)(u64{limit - offset}, granule - (current & (granule - 1)));
        if (!manager->IsValidAccessRange(current, amount, write)) {
            break;
        }
        const auto device = manager->GpuToCpuAddress(current);
        if (!device || (base && *device != *base + offset)) {
            break;
        }
        if (!base) {
            base = device;
        }
        offset += amount;
    }
    return offset;
}

void MemoryManager::WriteBlockImpl(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size, [[maybe_unused]] VideoCommon::CacheType which, bool unsafe) {
    if (OverlapsSyncpointShim(gpu_dest_addr, size)) {
        LOG_ERROR(HW_GPU, "Write to read-only syncpoint mapping at {:#x}, size={:#x}",
                  gpu_dest_addr, size);
    }
    auto just_advance = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset, std::size_t copy_amount) {
        src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
    };
    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
        }
        u8* physical = memory.GetPointer<u8>(dev_addr_base);
        std::memcpy(physical, src_buffer, copy_amount);
        src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
        }
        if (!IsBigPageContinuous(page_index)) [[unlikely]] {
            memory.WriteBlockUnsafe(dev_addr_base, src_buffer, copy_amount);
        } else {
            u8* physical = memory.GetPointer<u8>(dev_addr_base);
            std::memcpy(physical, src_buffer, copy_amount);
        }
        src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
    };
    auto write_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, just_advance, just_advance);
    };
    MemoryOperation(gpu_dest_addr, size, true, mapped_big, just_advance, write_short_pages);
}

void MemoryManager::WriteBlock(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size, VideoCommon::CacheType which) {
    WriteBlockImpl(gpu_dest_addr, src_buffer, size, which, false);
}

void MemoryManager::WriteBlockUnsafe(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size) {
    WriteBlockImpl(gpu_dest_addr, src_buffer, size, VideoCommon::CacheType::None, true);
}

void MemoryManager::WriteBlockCached(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size) {
    WriteBlockImpl(gpu_dest_addr, src_buffer, size, VideoCommon::CacheType::None, true);
    accumulator.Add(gpu_dest_addr, size);
}

void MemoryManager::FlushRegion(GPUVAddr gpu_addr, size_t size,
                                VideoCommon::CacheType which) const {
    auto do_nothing = [&]([[maybe_unused]] std::size_t page_index,
                          [[maybe_unused]] std::size_t offset,
                          [[maybe_unused]] std::size_t copy_amount) {};

    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
    };
    auto flush_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, do_nothing, do_nothing);
    };
    MemoryOperation(gpu_addr, size, true, mapped_big, do_nothing, flush_short_pages);
}

bool MemoryManager::IsMemoryDirty(GPUVAddr gpu_addr, size_t size,
                                  VideoCommon::CacheType which) const {
    bool result = false;
    auto do_nothing = [&]([[maybe_unused]] std::size_t page_index,
                          [[maybe_unused]] std::size_t offset,
                          [[maybe_unused]] std::size_t copy_amount) { return false; };

    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        result |= rasterizer->MustFlushRegion(dev_addr_base, copy_amount, which);
        return result;
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        result |= rasterizer->MustFlushRegion(dev_addr_base, copy_amount, which);
        return result;
    };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset,
                                 std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, do_nothing, do_nothing);
        return result;
    };
    MemoryOperation(gpu_addr, size, true, mapped_big, do_nothing, check_short_pages);
    return result;
}

size_t MemoryManager::MaxContinuousRange(GPUVAddr gpu_addr, size_t size) const {
    std::optional<DAddr> old_page_addr{};
    size_t range_so_far = 0;
    bool result{false};
    auto fail = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    std::size_t copy_amount) {
        result = true;
        return true;
    };
    auto short_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            result = true;
            return true;
        }
        range_so_far += copy_amount;
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto big_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            return true;
        }
        range_so_far += copy_amount;
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset,
                                 std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, short_check, fail, fail);
        return result;
    };
    MemoryOperation(gpu_addr, size, true, big_check, fail, check_short_pages);
    return range_so_far;
}

size_t MemoryManager::GetMemoryLayoutSize(GPUVAddr gpu_addr, size_t max_size) const {
    std::unique_lock<std::mutex> lock(guard);
    return kind_map.GetContinuousSizeFrom(gpu_addr);
}

void MemoryManager::InvalidateRegion(GPUVAddr gpu_addr, size_t size,
                                     VideoCommon::CacheType which) const {
    auto do_nothing = [&]([[maybe_unused]] std::size_t page_index,
                          [[maybe_unused]] std::size_t offset,
                          [[maybe_unused]] std::size_t copy_amount) {};

    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
    };
    auto invalidate_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, do_nothing, do_nothing);
    };
    MemoryOperation(gpu_addr, size, true, mapped_big, do_nothing, invalidate_short_pages);
}

void MemoryManager::CopyBlock(GPUVAddr gpu_dest_addr, GPUVAddr gpu_src_addr, std::size_t size,
                              VideoCommon::CacheType which) {
    Tegra::Memory::GpuGuestMemoryScoped<u8, GuestMemoryFlags::SafeReadWrite> data(
        *this, gpu_src_addr, size);
    data.SetAddressAndSize(gpu_dest_addr, size);
    FlushRegion(gpu_dest_addr, size, which);
}

bool MemoryManager::IsGranularRange(GPUVAddr gpu_addr, std::size_t size) const {
    if (GetEntry(gpu_addr, true) == EntryType::Mapped) [[likely]] {
        size_t page_index = gpu_addr >> big_page_bits;
        if (IsBigPageContinuous(page_index)) [[likely]] {
            const std::size_t page{(page_index & big_page_mask) + size};
            return page <= big_page_size;
        }
        const std::size_t page{(gpu_addr & Core::DEVICE_PAGEMASK) + size};
        return page <= Core::DEVICE_PAGESIZE;
    }
    if (GetEntry(gpu_addr, false) != EntryType::Mapped) {
        return false;
    }
    const std::size_t page{(gpu_addr & Core::DEVICE_PAGEMASK) + size};
    return page <= Core::DEVICE_PAGESIZE;
}

bool MemoryManager::IsContinuousRange(GPUVAddr gpu_addr, std::size_t size) const {
    std::optional<DAddr> old_page_addr{};
    bool result{true};
    auto fail = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    std::size_t copy_amount) {
        result = false;
        return true;
    };
    auto short_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            result = false;
            return true;
        }
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto big_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            result = false;
            return true;
        }
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset,
                                 std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, short_check, fail, fail);
        return !result;
    };
    MemoryOperation(gpu_addr, size, true, big_check, fail, check_short_pages);
    return result;
}

bool MemoryManager::IsFullyMappedRange(GPUVAddr gpu_addr, std::size_t size) const {
    bool result{true};
    auto fail = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    [[maybe_unused]] std::size_t copy_amount) {
        result = false;
        return true;
    };
    auto pass = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    [[maybe_unused]] std::size_t copy_amount) { return false; };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, pass, pass, fail);
        return !result;
    };
    MemoryOperation(gpu_addr, size, true, pass, fail, check_short_pages);
    return result;
}

boost::container::small_vector<std::pair<GPUVAddr, std::size_t>, 32>
MemoryManager::GetSubmappedRange(GPUVAddr gpu_addr, std::size_t size) const {
    boost::container::small_vector<std::pair<GPUVAddr, std::size_t>, 32> result{};
    GetSubmappedRangeImpl<true>(gpu_addr, size, result);
    return result;
}

template <bool is_gpu_address>
void MemoryManager::GetSubmappedRangeImpl(GPUVAddr gpu_addr, std::size_t size, boost::container::small_vector<std::pair<std::conditional_t<is_gpu_address, GPUVAddr, DAddr>, std::size_t>, 32>& result)
    const {
    std::optional<std::pair<std::conditional_t<is_gpu_address, GPUVAddr, DAddr>, std::size_t>> last_segment{};
    std::optional<DAddr> old_page_addr{};
    const auto split = [&last_segment, &result]([[maybe_unused]] std::size_t page_index,
                                                [[maybe_unused]] std::size_t offset,
                                                [[maybe_unused]] std::size_t copy_amount) {
        if (last_segment) {
            result.push_back(*last_segment);
            last_segment = std::nullopt;
        }
    };
    const auto extend_size_big = [this, &split, &old_page_addr,
                                  &last_segment](std::size_t page_index, std::size_t offset,
                                                 std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr) {
            if (*old_page_addr != dev_addr_base) {
                split(0, 0, 0);
            }
        }
        old_page_addr = {dev_addr_base + copy_amount};
        if (!last_segment) {
            if constexpr (is_gpu_address) {
                const GPUVAddr new_base_addr = (page_index << big_page_bits) + offset;
                last_segment = {new_base_addr, copy_amount};
            } else {
                last_segment = {dev_addr_base, copy_amount};
            }
        } else {
            last_segment->second += copy_amount;
        }
    };
    const auto extend_size_short = [this, &split, &old_page_addr,
                                    &last_segment](std::size_t page_index, std::size_t offset,
                                                   std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr) {
            if (*old_page_addr != dev_addr_base) {
                split(0, 0, 0);
            }
        }
        old_page_addr = {dev_addr_base + copy_amount};
        if (!last_segment) {
            if constexpr (is_gpu_address) {
                const GPUVAddr new_base_addr = (page_index << page_bits) + offset;
                last_segment = {new_base_addr, copy_amount};
            } else {
                last_segment = {dev_addr_base, copy_amount};
            }
        } else {
            last_segment->second += copy_amount;
        }
    };
    auto do_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, extend_size_short, split, split);
    };
    MemoryOperation(gpu_addr, size, true, extend_size_big, split, do_short_pages);
    split(0, 0, 0);
}

void MemoryManager::FlushCaching() {
    // Flush from the invalidate accumulator
    if (accumulator.InvalidateAll([this](GPUVAddr addr, size_t size) {
        GetSubmappedRangeImpl<false>(addr, size, page_stash2);
    })) {
        rasterizer->InnerInvalidation(page_stash2);
        page_stash2.clear();
    }
}

const u8* MemoryManager::GetSpan(const GPUVAddr src_addr, const std::size_t size) const {
    if (!IsContinuousRange(src_addr, size)) {
        return nullptr;
    }
    auto dev_addr = GpuToCpuAddress(src_addr);
    if (dev_addr) {
        return memory.GetSpan(*dev_addr, size);
    }
    return nullptr;
}

u8* MemoryManager::GetSpan(const GPUVAddr src_addr, const std::size_t size) {
    if (!IsContinuousRange(src_addr, size)) {
        return nullptr;
    }
    auto dev_addr = GpuToCpuAddress(src_addr);
    if (dev_addr) {
        return memory.GetSpan(*dev_addr, size);
    }
    return nullptr;
}

} // namespace Tegra
