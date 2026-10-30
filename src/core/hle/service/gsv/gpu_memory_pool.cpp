// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include "common/alignment.h"
#include "common/scope_exit.h"
#include "common/swap.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/gpu_address_space_pool.h"
#include "core/hle/service/gsv/gpu_memory_pool.h"
#include "core/hle/service/gsv/graphics_device_table.h"
#include "video_core/memory_manager.h"

namespace Service::GSV {
namespace {
constexpr Result InvalidArgument{0x100f3};
constexpr Result InvalidMemory{0x280f3};
constexpr Result OutOfMemory{0x80f3};

struct SparseMapRequest {
    u64_le address;
    u64_le size;
    u8 permission;
    u8 cacheability;
    u8 reserved_12;
    u8 kind;
    u8 incompressible_kind;
    u8 raw_texture;
    u16_le reserved_16;
    u32_le memory_handle;
    u32_le reserved_1c;
    u64_le offset;
};
static_assert(sizeof(SparseMapRequest) == 40);
static_assert(offsetof(SparseMapRequest, memory_handle) == 24);
static_assert(offsetof(SparseMapRequest, offset) == 32);

Result SelectMemoryKind(u8* out_kind, Tegra::PTEKind* out_layout, u8 requested, u8 fallback) {
    // Native kinds use the Turing MMU encoding, not the legacy enum's numeric values.
    // https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_mmu.ref.txt
    constexpr std::array layouts{Tegra::PTEKind::PITCH,        Tegra::PTEKind::Z16,
                                 Tegra::PTEKind::S8,           Tegra::PTEKind::S8Z24,
                                 Tegra::PTEKind::ZF32_X24S8,   Tegra::PTEKind::Z24S8,
                                 Tegra::PTEKind::GENERIC_16BX2};
    R_UNLESS(requested <= 6 || (requested >= 8 && requested <= 14), InvalidArgument);
    // Host renderers store uncompressed images, so expose the caller's uncompressed fallback.
    const u8 selected = requested >= 8 ? fallback : requested;
    R_UNLESS(selected < layouts.size(), Kernel::ResultNotImplemented);
    *out_kind = selected;
    *out_layout = layouts[selected];
    R_SUCCEED();
}
} // namespace

GpuMemoryPool::GpuMemoryPool(Kernel::KernelCore& kernel, Kernel::KProcess* process,
                             std::shared_ptr<GraphicsDeviceTable> devices)
    : m_kernel{kernel}, m_process{kernel, process}, m_devices{std::move(devices)} {}

GpuMemoryPool::~GpuMemoryPool() {
    while (!m_mappings.empty()) {
        RemoveMapping(m_mappings.begin(), false);
    }
    for (const auto& [key, region] : m_sparse_regions) {
        region.manager->Unmap(key.second, region.size);
        m_devices->ReleaseMemory(region.device_id);
    }
    while (!m_memory.empty()) {
        const auto it = m_memory.begin();
        const auto result = ReleaseMemory(it);
        if (result.IsError()) {
            LOG_ERROR(Service, "GPU memory teardown could not release handle={:#x}: {:#x}",
                      it->first, result.raw);
            m_devices->ReleaseMemory(it->second.device_id);
            m_memory.erase(it);
        }
    }
    if (m_asid) {
        m_backing->UnregisterProcess(*m_asid);
    }
}

Result GpuMemoryPool::Register(u32* out_handle, u64 address, u64 last_address, u8 flags,
                               u8 device_id) {
    R_UNLESS(Common::IsAligned(address, Kernel::PageSize) && last_address >= address &&
                 (last_address >> 59) == 0 &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize) && flags <= 1 &&
                 device_id < GraphicsDeviceTable::Capacity,
             InvalidArgument);
    const std::scoped_lock lock{m_mutex};
    R_UNLESS(m_memory.size() < 0xffff, OutOfMemory);
    R_TRY(m_devices->RetainMemory(device_id));
    bool retained{};
    SCOPE_EXIT {
        if (!retained)
            m_devices->ReleaseMemory(device_id);
    };

    try {
        u32 handle;
        u32 counter = m_next_handle;
        do {
            counter = (counter + 1) & 0xffffff;
            if (counter == 0) {
                counter = 1;
            }
            handle = 0x01000000 | counter;
        } while (m_memory.contains(handle));
        auto it = m_memory.try_emplace(handle, m_kernel).first;
        auto& memory = it->second;
        memory.address = address;
        memory.device_id = device_id;
        SCOPE_EXIT {
            if (!retained) {
                if (memory.pinned_size != 0) {
                    // An unsuccessful registration owns its partial pins until rollback finishes.
                    const auto result = m_process->GetPageTable().UnlockGpuSharableMemory(
                        address, memory.pinned_size);
                    if (result.IsError()) {
                        // Preserve ownership for context teardown if the kernel cannot allocate
                        // the block metadata needed to split an adjacent pinned range.
                        retained = true;
                        LOG_ERROR(Service, "GPU memory rollback retained handle={:#x}: {:#x}",
                                  handle, result.raw);
                        return;
                    }
                }
                m_memory.erase(it);
            }
        };
        const u64 size = last_address - address + 1;
        while (memory.pinned_size != size) {
            Kernel::Svc::lp64::PhysicalMemoryInfo info;
            R_TRY(m_process->GetPageTable().LockGpuSharableMemory(
                &info, address + memory.pinned_size, size - memory.pinned_size));
            memory.pinned_size += info.size;
            memory.chunks.push_back(info);
        }
        *out_handle = handle;
        m_next_handle = counter;
        retained = true;
        LOG_DEBUG(Service, "GPU memory registered: handle={:#x}, size={:#x}, chunks={}, device={}",
                  handle, size, memory.chunks.size(), device_id);
        R_SUCCEED();
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
}

Result GpuMemoryPool::Unregister(u32 handle, Kernel::KEvent* completion) {
    const std::scoped_lock lock{m_mutex};
    const auto it = m_memory.find(handle);
    R_UNLESS(it != m_memory.end() && !it->second.release_pending, InvalidMemory);
    if (it->second.mapping_count != 0 || it->second.reference_count != 0) {
        it->second.release_pending = true;
        it->second.release_event.SetObject(completion);
        R_SUCCEED();
    }
    R_RETURN(ReleaseMemory(it, completion));
}

Result GpuMemoryPool::RetainReference(u32 handle, u64 address, u64 size, u64 offset) {
    const std::scoped_lock lock{m_mutex};
    const auto it = m_memory.find(handle);
    R_UNLESS(it != m_memory.end() && !it->second.release_pending, InvalidMemory);
    auto& memory = it->second;
    R_UNLESS(size != 0 && offset <= memory.pinned_size && size <= memory.pinned_size - offset &&
                 address >= memory.address && address - memory.address == offset,
             InvalidArgument);
    R_UNLESS(memory.reference_count < std::numeric_limits<size_t>::max(), OutOfMemory);
    ++memory.reference_count;
    R_SUCCEED();
}

Result GpuMemoryPool::ReleaseReference(u32 handle) {
    const std::scoped_lock lock{m_mutex};
    const auto it = m_memory.find(handle);
    R_UNLESS(it != m_memory.end() && it->second.reference_count != 0, InvalidMemory);
    --it->second.reference_count;
    if (it->second.release_pending && it->second.mapping_count == 0 &&
        it->second.reference_count == 0) {
        const auto result = ReleaseMemory(it);
        if (result.IsError()) {
            // Keep the caller's reference if releasing physical ownership could not finish.
            ++it->second.reference_count;
            R_RETURN(result);
        }
    }
    R_SUCCEED();
}

Result GpuMemoryPool::ReleaseMemory(std::map<u32, Memory>::iterator it,
                                    Kernel::KEvent* completion) {
    const u32 handle = it->first;
    const auto& memory = it->second;
    Kernel::KScopedAutoObject event{m_kernel, completion ? completion
                                                         : memory.release_event.GetPointerUnsafe()};
    if (memory.device_address != 0) {
        m_backing->Unmap(memory.device_address, memory.pinned_size);
        m_backing->Free(memory.device_address, memory.pinned_size);
        it->second.device_address = 0;
    }
    R_TRY(m_process->GetPageTable().UnlockGpuSharableMemory(memory.address, memory.pinned_size));
    m_devices->ReleaseMemory(memory.device_id);
    m_memory.erase(it);
    LOG_DEBUG(Service, "GPU memory unregistered: handle={:#x}", handle);
    if (event.IsNotNull()) {
        // A disappearing waiter must not prevent physical resources from being released.
        const auto result = event->Signal(m_kernel);
        if (result.IsError()) {
            LOG_DEBUG(Service, "GPU release event is no longer available: {:#x}", result.raw);
        }
    }
    R_SUCCEED();
}

Result GpuMemoryPool::Map(u8* out_kind, GpuAddressSpacePool& address_spaces,
                          Tegra::MaxwellDeviceMemoryManager& backing, u32 address_space,
                          u64 gpu_address, u64 last_address, u32 memory_handle, u64 offset,
                          MapOptions options) {
    const std::scoped_lock lock{m_mutex};
    R_RETURN(MapImpl(out_kind, address_spaces, backing, address_space, gpu_address, last_address,
                     memory_handle, offset, options, false));
}

Result GpuMemoryPool::MapSparseMemory(u8* out_kind, GpuAddressSpacePool& address_spaces,
                                      Tegra::MaxwellDeviceMemoryManager& backing, u32 address_space,
                                      u64 gpu_address, u64 last_address, u32 memory_handle,
                                      u64 offset, MapOptions options) {
    const std::scoped_lock lock{m_mutex};
    R_RETURN(MapImpl(out_kind, address_spaces, backing, address_space, gpu_address, last_address,
                     memory_handle, offset, options, true));
}

Result GpuMemoryPool::ModifyKind(u8* out_kind, GpuAddressSpacePool& address_spaces,
                                 u32 address_space, u64 gpu_address, u64 last_address, u8 requested,
                                 u8 fallback) {
    R_UNLESS(last_address >= gpu_address && last_address < (1ULL << 40) &&
                 Common::IsAligned(gpu_address, Kernel::PageSize) &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize),
             InvalidArgument);
    const std::scoped_lock lock{m_mutex};
    u8 selected{};
    Tegra::PTEKind kind{};
    R_TRY(SelectMemoryKind(&selected, &kind, requested, fallback));
    const auto manager = address_spaces.Acquire(address_space);
    R_UNLESS(manager, InvalidMemory);
    u64 first, last;
    R_TRY(address_spaces.QueryPartitionRange(&first, &last, address_space,
                                             GpuAddressSpacePool::Partition::User));
    R_UNLESS(gpu_address >= first && last_address <= last, InvalidArgument);
    const u64 end = last_address + 1;
    for (u64 cursor = gpu_address; cursor < end;) {
        auto it = m_mappings.upper_bound({address_space, cursor});
        R_UNLESS(it != m_mappings.begin(), InvalidMemory);
        --it;
        R_UNLESS(it->first.first == address_space && it->second.manager == manager &&
                     cursor >= it->first.second && cursor < it->first.second + it->second.size,
                 InvalidMemory);
        cursor = (std::min)(end, it->first.second + it->second.size);
    }
    try {
        R_UNLESS(manager->ModifyPageKindChecked(gpu_address, end - gpu_address, kind),
                 InvalidMemory);
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
    *out_kind = selected;
    LOG_DEBUG(Service,
              "Native GPU memory kind modified: as={:#x}, gpu={:#x}, size={:#x}, requested={}, "
              "selected={}",
              address_space, gpu_address, end - gpu_address, requested, selected);
    R_SUCCEED();
}

Result GpuMemoryPool::MapImpl(u8* out_kind, GpuAddressSpacePool& address_spaces,
                              Tegra::MaxwellDeviceMemoryManager& backing, u32 address_space,
                              u64 gpu_address, u64 last_address, u32 memory_handle, u64 offset,
                              MapOptions options, bool sparse_map) {
    R_UNLESS(last_address >= gpu_address && last_address < (1ULL << 40) &&
                 Common::IsAligned(gpu_address, Kernel::PageSize) &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize) &&
                 Common::IsAligned(offset, Kernel::PageSize) && options.permission <= 1 &&
                 options.cacheability <= 1 && options.raw_texture <= 1 && options.sync <= 1 &&
                 options.device_id < GraphicsDeviceTable::Capacity,
             InvalidArgument);
    R_UNLESS(options.permission == 0 && options.raw_texture == 0, Kernel::ResultNotImplemented);
    u8 selected_kind{};
    Tegra::PTEKind kind{};
    R_TRY(SelectMemoryKind(&selected_kind, &kind, options.kind, options.incompressible_kind));
    const auto manager = address_spaces.Acquire(address_space);
    R_UNLESS(manager, InvalidMemory);
    u64 first, last;
    R_TRY(address_spaces.QueryPartitionRange(&first, &last, address_space,
                                             GpuAddressSpacePool::Partition::User));
    R_UNLESS(gpu_address >= first && last_address <= last, InvalidArgument);
    const u64 size = last_address - gpu_address + 1;
    const auto memory = m_memory.find(memory_handle);
    R_UNLESS(memory != m_memory.end() && !memory->second.release_pending, InvalidMemory);
    R_UNLESS(memory->second.device_id == options.device_id &&
                 offset <= memory->second.pinned_size &&
                 size <= memory->second.pinned_size - offset,
             InvalidArgument);
    R_UNLESS(!m_backing || m_backing == &backing, Kernel::ResultInvalidState);
    const auto sparse = FindSparseRegion(address_space, gpu_address);
    R_UNLESS(sparse_map == (sparse != m_sparse_regions.end()), Result{0x200f3});
    if (sparse != m_sparse_regions.end()) {
        const auto& region = sparse->second;
        R_UNLESS(last_address < sparse->first.second + region.size &&
                     options.device_id == region.device_id,
                 InvalidArgument);
        const u64 granularity = 1ULL << region.page_bits;
        R_UNLESS(Common::IsAligned(gpu_address, granularity) &&
                     Common::IsAligned(size, granularity) && Common::IsAligned(offset, granularity),
                 InvalidArgument);
    }
    const auto key = std::pair{address_space, gpu_address};
    auto next = m_mappings.lower_bound(key);
    R_UNLESS(next == m_mappings.end() || next->first.first != address_space ||
                 next->first.second > last_address,
             Result{0x200f3});
    if (next != m_mappings.begin()) {
        const auto previous = std::prev(next);
        R_UNLESS(previous->first.first != address_space ||
                     previous->first.second + previous->second.size <= gpu_address,
                 Result{0x200f3});
    }

    try {
        auto it = m_mappings.try_emplace(key).first;
        bool keep{};
        bool created_backing{};
        SCOPE_EXIT {
            if (!keep) {
                if (created_backing) {
                    backing.Unmap(memory->second.device_address, memory->second.pinned_size);
                    backing.Free(memory->second.device_address, memory->second.pinned_size);
                    memory->second.device_address = 0;
                }
                m_mappings.erase(it);
            }
        };
        if (!m_asid) {
            m_asid = backing.RegisterProcess(&m_process->GetMemory());
            m_backing = &backing;
        }
        if (memory->second.device_address == 0) {
            // Aliases and address spaces share one translation of the retained CPU allocation.
            // The intermediate device address space must not charge its RAM size for every alias.
            memory->second.device_address = backing.Allocate(memory->second.pinned_size);
            R_UNLESS(memory->second.device_address != 0, OutOfMemory);
            created_backing = true;
            backing.Map(memory->second.device_address, memory->second.address,
                        memory->second.pinned_size, *m_asid, true);
        }
        const auto device_address = memory->second.device_address + offset;
        const auto retain_mapping = [&] {
            it->second = Mapping{manager, size, memory_handle};
            ++memory->second.mapping_count;
            keep = true;
        };
        try {
            const bool mapped =
                sparse != m_sparse_regions.end()
                    ? manager->MapIntoSparse(gpu_address, device_address, size, kind)
                    : manager->MapIfFree(gpu_address, device_address, size, kind);
            R_UNLESS(mapped, Result{0x200f3});
        } catch (const std::bad_alloc&) {
            // GMMU metadata allocation can fail after installing some page entries. Keep their
            // backing owned until unmap/teardown rather than freeing a still-referenced address.
            retain_mapping();
            throw;
        }
        retain_mapping();
        *out_kind = selected_kind;
        LOG_DEBUG(
            Service,
            "Native GPU memory mapped: as={:#x}, gpu={:#x}, size={:#x}, memory={:#x}, kind={}",
            address_space, gpu_address, size, memory_handle, selected_kind);
        R_SUCCEED();
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
}

void GpuMemoryPool::RemoveMapping(std::map<std::pair<u32, u64>, Mapping>::iterator it,
                                  bool restore_sparse) {
    const auto& mapping = it->second;
    mapping.manager->Unmap(it->first.second, mapping.size,
                           restore_sparse && FindSparseRegion(it->first.first, it->first.second) !=
                                                 m_sparse_regions.end());
    const auto memory = m_memory.find(mapping.memory_handle);
    ASSERT(memory != m_memory.end() && memory->second.mapping_count != 0);
    --memory->second.mapping_count;
    m_mappings.erase(it);
    if (memory->second.release_pending && memory->second.mapping_count == 0 &&
        memory->second.reference_count == 0) {
        const auto result = ReleaseMemory(memory);
        if (result.IsError()) {
            LOG_ERROR(Service, "Deferred GPU memory release failed: {:#x}", result.raw);
        }
    }
}

Result GpuMemoryPool::Unmap(u32 address_space, u64 gpu_address, u64 last_address,
                            u64* out_unmapped) {
    const std::scoped_lock lock{m_mutex};
    R_RETURN(UnmapImpl(address_space, gpu_address, last_address, out_unmapped, false));
}

Result GpuMemoryPool::UnmapSparseMemory(u32 address_space, u64 gpu_address, u64 last_address,
                                        u64* out_unmapped) {
    const std::scoped_lock lock{m_mutex};
    R_RETURN(UnmapImpl(address_space, gpu_address, last_address, out_unmapped, true));
}

Result GpuMemoryPool::UnmapImpl(u32 address_space, u64 gpu_address, u64 last_address,
                                u64* out_unmapped, bool sparse_unmap) {
    R_UNLESS(last_address >= gpu_address && last_address < (1ULL << 40) &&
                 Common::IsAligned(gpu_address, Kernel::PageSize) &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize),
             InvalidArgument);
    auto containing = [&](u64 address) {
        auto it = m_mappings.upper_bound({address_space, address});
        if (it == m_mappings.begin())
            return m_mappings.end();
        --it;
        return it->first.first == address_space && address < it->first.second + it->second.size
                   ? it
                   : m_mappings.end();
    };
    const u64 end = last_address + 1;
    // Validate coverage before removing anything, including requests crossing adjacent mappings.
    for (u64 cursor = gpu_address; cursor < end;) {
        const auto it = containing(cursor);
        const auto sparse = FindSparseRegion(address_space, cursor);
        if (sparse != m_sparse_regions.end()) {
            const u64 region_end = sparse->first.second + sparse->second.size;
            const u64 removed_end = std::min(end, region_end);
            const u64 granularity = 1ULL << sparse->second.page_bits;
            R_UNLESS(!sparse_unmap || (Common::IsAligned(cursor, granularity) &&
                                       Common::IsAligned(removed_end, granularity)),
                     InvalidArgument);
            cursor = removed_end;
        } else {
            R_UNLESS(!sparse_unmap, InvalidMemory);
            R_UNLESS(it != m_mappings.end(), InvalidMemory);
            cursor = std::min(end, it->first.second + it->second.size);
        }
    }
    u64 unmapped{};
    try {
        for (u64 cursor = gpu_address; cursor < end;) {
            const auto sparse = FindSparseRegion(address_space, cursor);
            if (!sparse_unmap && sparse != m_sparse_regions.end()) {
                // NormalUnmap walks ordinary allocations; sparse parents and their resident
                // children belong to ExecuteSparseMapRequests instead.
                cursor = std::min(end, sparse->first.second + sparse->second.size);
                continue;
            }
            auto it = containing(cursor);
            if (it == m_mappings.end()) {
                ASSERT(sparse != m_sparse_regions.end());
                auto next = m_mappings.lower_bound({address_space, cursor});
                cursor = std::min(end, sparse->first.second + sparse->second.size);
                if (next != m_mappings.end() && next->first.first == address_space) {
                    cursor = std::min(cursor, next->first.second);
                }
                continue;
            }
            const auto start = it->first.second;
            const auto mapping_end = start + it->second.size;
            const auto removed_end = std::min(end, mapping_end);
            auto* manager = it->second.manager.get();
            if (cursor == start && removed_end == mapping_end) {
                RemoveMapping(it);
            } else {
                auto right = m_mappings.end();
                bool keep_right{};
                SCOPE_EXIT {
                    if (right != m_mappings.end() && !keep_right) {
                        m_mappings.erase(right);
                    }
                };
                if (cursor != start && removed_end < mapping_end) {
                    auto mapping = it->second;
                    mapping.size = mapping_end - removed_end;
                    right = m_mappings
                                .emplace(std::pair{address_space, removed_end}, std::move(mapping))
                                .first;
                }
                // Keep the original ownership record until the GMMU operation finishes. A
                // metadata allocation failure can leave some PTEs installed or removed.
                manager->Unmap(cursor, removed_end - cursor,
                               FindSparseRegion(address_space, cursor) != m_sparse_regions.end());
                if (cursor == start) {
                    auto node = m_mappings.extract(it);
                    node.key().second = removed_end;
                    node.mapped().size = mapping_end - removed_end;
                    m_mappings.insert(std::move(node));
                } else {
                    if (right != m_mappings.end()) {
                        ++m_memory.at(it->second.memory_handle).mapping_count;
                    }
                    it->second.size = cursor - start;
                }
                keep_right = true;
            }
            unmapped += removed_end - cursor;
            cursor = removed_end;
        }
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
    if (out_unmapped) {
        *out_unmapped = unmapped;
    }
    R_SUCCEED();
}

GpuMemoryPool::SparseRegions::iterator GpuMemoryPool::FindSparseRegion(u32 address_space,
                                                                       u64 address) {
    auto it = m_sparse_regions.upper_bound({address_space, address});
    if (it == m_sparse_regions.begin()) {
        return m_sparse_regions.end();
    }
    --it;
    return it->first.first == address_space && address < it->first.second + it->second.size
               ? it
               : m_sparse_regions.end();
}

Result GpuMemoryPool::ExecuteSparseMapRequests(u64* out_processed, std::span<u8> out_kinds,
                                               GpuAddressSpacePool& address_spaces,
                                               Tegra::MaxwellDeviceMemoryManager& backing,
                                               u32 address_space, std::span<const u8> bytes,
                                               u64 version, u64 stride, u8 sync,
                                               bool* out_accepted) {
    if (out_accepted) {
        *out_accepted = false;
    }
    R_UNLESS(version == 1 && stride == sizeof(SparseMapRequest) && sync <= 1 &&
                 bytes.size() % sizeof(SparseMapRequest) == 0,
             InvalidArgument);
    if (bytes.empty()) {
        *out_processed = 0;
        if (out_accepted) {
            *out_accepted = true;
        }
        R_SUCCEED();
    }
    try {
        // Snapshot every request before writing kinds; guest input and output buffers may alias.
        std::vector<SparseMapRequest> requests(bytes.size() / sizeof(SparseMapRequest));
        std::memcpy(requests.data(), bytes.data(), bytes.size());
        for (const auto& request : requests) {
            const u64 address = request.address;
            const u64 size = request.size;
            const u64 offset = request.offset;
            R_UNLESS(address < (1ULL << 40) && size != 0 && size <= (1ULL << 40) - address &&
                         Common::IsAligned(address, Kernel::PageSize) &&
                         Common::IsAligned(size, Kernel::PageSize) && request.reserved_12 == 0 &&
                         request.reserved_16 == 0 && request.reserved_1c == 0,
                     InvalidArgument);
            if (request.memory_handle == 0) {
                R_UNLESS(request.permission == 0 && request.cacheability == 0 &&
                             request.kind == 0 && request.incompressible_kind == 0 &&
                             request.raw_texture == 0 && offset == 0,
                         InvalidArgument);
            } else {
                R_UNLESS(request.permission <= 1 && request.cacheability <= 1 &&
                             request.raw_texture <= 1 && offset < (1ULL << 37) &&
                             size <= (1ULL << 37) - offset &&
                             Common::IsAligned(offset, Kernel::PageSize),
                         InvalidArgument);
                R_UNLESS(request.permission == 0 && request.raw_texture == 0,
                         Kernel::ResultNotImplemented);
                u8 selected{};
                Tegra::PTEKind layout{};
                R_TRY(SelectMemoryKind(&selected, &layout, request.kind,
                                       request.incompressible_kind));
            }
        }

        const auto manager = address_spaces.Acquire(address_space);
        R_UNLESS(manager, InvalidMemory);
        const std::scoped_lock lock{m_mutex};
        R_UNLESS(!m_backing || m_backing == &backing, Kernel::ResultInvalidState);
        const auto region = FindSparseRegion(address_space, requests.front().address);
        R_UNLESS(region != m_sparse_regions.end() && region->second.manager == manager,
                 InvalidMemory);
        const auto device = region->second.device_id;
        const u64 granularity = 1ULL << region->second.page_bits;
        // Native batches operate on one sparse parent. Validate all handles and ranges before
        // replacing any residents, including handles used again later in the same batch.
        for (const auto& request : requests) {
            const u64 address = request.address;
            const u64 size = request.size;
            const u64 offset = request.offset;
            R_UNLESS(address >= region->first.second &&
                         address + size <= region->first.second + region->second.size &&
                         Common::IsAligned(address, granularity) &&
                         Common::IsAligned(size, granularity) &&
                         Common::IsAligned(offset, granularity),
                     InvalidArgument);
            if (request.memory_handle != 0) {
                const auto memory = m_memory.find(request.memory_handle);
                R_UNLESS(memory != m_memory.end() && !memory->second.release_pending,
                         InvalidMemory);
                R_UNLESS(memory->second.device_id == device &&
                             offset <= memory->second.pinned_size &&
                             size <= memory->second.pinned_size - offset,
                         InvalidArgument);
            }
        }

        *out_processed = 0;
        if (out_accepted) {
            *out_accepted = true;
        }
        for (const auto& request : requests) {
            const u64 address = request.address;
            const u64 last = address + request.size - 1;
            u8 kind{};
            R_TRY(UnmapImpl(address_space, address, last, nullptr, true));
            if (request.memory_handle != 0) {
                const MapOptions options{request.permission,
                                         request.cacheability,
                                         request.kind,
                                         request.incompressible_kind,
                                         request.raw_texture,
                                         device,
                                         sync};
                R_TRY(MapImpl(&kind, address_spaces, backing, address_space, address, last,
                              request.memory_handle, request.offset, options, true));
            }
            if (*out_processed < out_kinds.size()) {
                out_kinds[*out_processed] = kind;
            }
            ++*out_processed;
        }
        LOG_DEBUG(Service, "Native GPU sparse batch completed: as={:#x}, requests={}",
                  address_space, *out_processed);
        R_SUCCEED();
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
}

Result GpuMemoryPool::CreateSparseRegion(GpuAddressSpacePool& address_spaces, u32 address_space,
                                         u64 gpu_address, u64 last_address, u8 page_bits,
                                         u8 alloc_mode, u8 device_id) {
    R_UNLESS((page_bits == 12 || page_bits == 16) && alloc_mode <= 1 &&
                 device_id < GraphicsDeviceTable::Capacity && last_address >= gpu_address &&
                 last_address < (1ULL << 40),
             InvalidArgument);
    const u64 size = last_address - gpu_address + 1;
    const u64 granularity = 1ULL << page_bits;
    R_UNLESS(Common::IsAligned(gpu_address, granularity) && Common::IsAligned(size, granularity),
             InvalidArgument);
    const auto manager = address_spaces.Acquire(address_space);
    R_UNLESS(manager, InvalidMemory);
    u64 first, last;
    R_TRY(address_spaces.QueryPartitionRange(&first, &last, address_space,
                                             GpuAddressSpacePool::Partition::User));
    R_UNLESS(gpu_address >= first && last_address <= last, InvalidArgument);
    const std::scoped_lock lock{m_mutex};
    const auto overlaps = [&](const auto& ranges) {
        const auto next = ranges.lower_bound(std::pair{address_space, gpu_address});
        if (next != ranges.end() && next->first.first == address_space &&
            next->first.second <= last_address) {
            return true;
        }
        if (next != ranges.begin()) {
            const auto previous = std::prev(next);
            return previous->first.first == address_space &&
                   previous->first.second + previous->second.size > gpu_address;
        }
        return false;
    };
    R_UNLESS(!overlaps(m_sparse_regions) && !overlaps(m_mappings), Result{0x200f3});
    R_TRY(m_devices->RetainMemory(device_id));
    bool keep{};
    SCOPE_EXIT {
        if (!keep) {
            m_devices->ReleaseMemory(device_id);
        }
    };
    try {
        const auto [it, inserted] = m_sparse_regions.try_emplace(
            std::pair{address_space, gpu_address},
            SparseRegion{manager, size, page_bits, alloc_mode, device_id});
        R_UNLESS(inserted, Result{0x200f3});
        SCOPE_EXIT {
            if (!keep) {
                m_sparse_regions.erase(it);
            }
        };
        try {
            R_UNLESS(manager->ReserveSparseIfFree(gpu_address, size), Result{0x200f3});
        } catch (const std::bad_alloc&) {
            // Retain ownership if page-table metadata allocation failed after installing entries.
            keep = true;
            throw;
        }
        keep = true;
        LOG_DEBUG(Service,
                  "Native GPU sparse region created: as={:#x}, gpu={:#x}, size={:#x}, "
                  "page_bits={}, alloc_mode={}, device={}",
                  address_space, gpu_address, size, page_bits, alloc_mode, device_id);
        R_SUCCEED();
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
}

Result GpuMemoryPool::DestroySparseRegion(u32 address_space, u64 gpu_address, u64 last_address) {
    R_UNLESS(last_address >= gpu_address && last_address < (1ULL << 40) &&
                 Common::IsAligned(gpu_address, Kernel::PageSize) &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize),
             InvalidArgument);
    const std::scoped_lock lock{m_mutex};
    const auto region = m_sparse_regions.find({address_space, gpu_address});
    R_UNLESS(region != m_sparse_regions.end() &&
                 region->second.size == last_address - gpu_address + 1,
             InvalidMemory);
    try {
        auto it = m_mappings.lower_bound({address_space, gpu_address});
        while (it != m_mappings.end() && it->first.first == address_space &&
               it->first.second <= last_address) {
            RemoveMapping(it++, false);
        }
        region->second.manager->Unmap(gpu_address, region->second.size);
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
    m_devices->ReleaseMemory(region->second.device_id);
    m_sparse_regions.erase(region);
    R_SUCCEED();
}

} // namespace Service::GSV
