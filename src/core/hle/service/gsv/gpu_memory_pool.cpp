// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <new>
#include "common/alignment.h"
#include "common/scope_exit.h"
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
} // namespace

GpuMemoryPool::GpuMemoryPool(Kernel::KernelCore& kernel, Kernel::KProcess* process,
                             std::shared_ptr<GraphicsDeviceTable> devices)
    : m_kernel{kernel}, m_process{kernel, process}, m_devices{std::move(devices)} {}

GpuMemoryPool::~GpuMemoryPool() {
    while (!m_mappings.empty()) {
        RemoveMapping(m_mappings.begin());
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
    R_UNLESS(last_address >= gpu_address && last_address < (1ULL << 40) &&
                 Common::IsAligned(gpu_address, Kernel::PageSize) &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize) &&
                 Common::IsAligned(offset, Kernel::PageSize) && options.permission <= 1 &&
                 options.cacheability <= 1 && options.raw_texture <= 1 && options.sync <= 1 &&
                 options.device_id < GraphicsDeviceTable::Capacity,
             InvalidArgument);
    // Read-only PTEs and compressed/raw texture mappings need their own access/layout support.
    R_UNLESS(options.permission == 0 && options.raw_texture == 0 &&
                 options.kind == options.compressed_kind &&
                 (options.kind == 0 || options.kind == 6),
             Kernel::ResultNotImplemented);
    const auto manager = address_spaces.Acquire(address_space);
    R_UNLESS(manager, InvalidMemory);
    u64 first, last;
    R_TRY(address_spaces.QueryPartitionRange(&first, &last, address_space,
                                             GpuAddressSpacePool::Partition::User));
    R_UNLESS(gpu_address >= first && last_address <= last, InvalidArgument);
    const u64 size = last_address - gpu_address + 1;
    const std::scoped_lock lock{m_mutex};
    const auto memory = m_memory.find(memory_handle);
    R_UNLESS(memory != m_memory.end() && !memory->second.release_pending, InvalidMemory);
    R_UNLESS(memory->second.device_id == options.device_id &&
                 offset <= memory->second.pinned_size &&
                 size <= memory->second.pinned_size - offset,
             InvalidArgument);
    R_UNLESS(!m_backing || m_backing == &backing, Kernel::ResultInvalidState);
    // Include sparse reservations and driver-owned apertures in the collision check below.
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
        // Native generic memory (6) is not the legacy Z16 kind at that numeric value.
        const auto kind = options.kind == 0 ? Tegra::PTEKind::PITCH : Tegra::PTEKind::GENERIC_16BX2;
        const auto retain_mapping = [&] {
            it->second = Mapping{manager, size, memory_handle};
            ++memory->second.mapping_count;
            keep = true;
        };
        try {
            R_UNLESS(manager->MapIfFree(gpu_address, device_address, size, kind), Result{0x200f3});
        } catch (const std::bad_alloc&) {
            // GMMU metadata allocation can fail after installing some page entries. Keep their
            // backing owned until unmap/teardown rather than freeing a still-referenced address.
            retain_mapping();
            throw;
        }
        retain_mapping();
        *out_kind = options.kind;
        LOG_DEBUG(
            Service,
            "Native GPU memory mapped: as={:#x}, gpu={:#x}, size={:#x}, memory={:#x}, kind={}",
            address_space, gpu_address, size, memory_handle, options.kind);
        R_SUCCEED();
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
}

void GpuMemoryPool::RemoveMapping(std::map<std::pair<u32, u64>, Mapping>::iterator it) {
    const auto& mapping = it->second;
    mapping.manager->Unmap(it->first.second, mapping.size);
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

Result GpuMemoryPool::Unmap(u32 address_space, u64 gpu_address, u64 last_address) {
    R_UNLESS(last_address >= gpu_address && last_address < (1ULL << 40) &&
                 Common::IsAligned(gpu_address, Kernel::PageSize) &&
                 Common::IsAligned(last_address + 1, Kernel::PageSize),
             InvalidArgument);
    const std::scoped_lock lock{m_mutex};
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
        R_UNLESS(it != m_mappings.end(), InvalidMemory);
        cursor = std::min(end, it->first.second + it->second.size);
    }
    try {
        for (u64 cursor = gpu_address; cursor < end;) {
            auto it = containing(cursor);
            const auto start = it->first.second;
            const auto mapping_end = start + it->second.size;
            const auto removed_end = std::min(end, mapping_end);
            auto* manager = it->second.manager.get();
            if (cursor == start && removed_end == mapping_end) {
                RemoveMapping(it);
            } else {
                if (cursor == start) {
                    auto node = m_mappings.extract(it);
                    node.key().second = removed_end;
                    node.mapped().size = mapping_end - removed_end;
                    m_mappings.insert(std::move(node));
                } else {
                    if (removed_end < mapping_end) {
                        // A middle split is the sole mapping in this request. Allocate its new
                        // record before changing the original range or the GPU page table.
                        auto right = it->second;
                        right.size = mapping_end - removed_end;
                        m_mappings.emplace(std::pair{address_space, removed_end}, std::move(right));
                        ++m_memory.at(it->second.memory_handle).mapping_count;
                    }
                    it->second.size = cursor - start;
                }
                manager->Unmap(cursor, removed_end - cursor);
            }
            cursor = removed_end;
        }
    } catch (const std::bad_alloc&) {
        R_RETURN(OutOfMemory);
    }
    R_SUCCEED();
}

} // namespace Service::GSV
