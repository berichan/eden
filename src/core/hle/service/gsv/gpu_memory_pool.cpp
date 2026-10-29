// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <new>
#include "common/alignment.h"
#include "common/scope_exit.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/gpu_memory_pool.h"
#include "core/hle/service/gsv/graphics_device_table.h"

namespace Service::GSV {
namespace {
constexpr Result InvalidArgument{0x100f3};
constexpr Result InvalidMemory{0x280f3};
constexpr Result OutOfMemory{0x80f3};
} // namespace

GpuMemoryPool::GpuMemoryPool(Kernel::KernelCore& kernel, Kernel::KProcess* process,
                             std::shared_ptr<GraphicsDeviceTable> devices)
    : m_process{kernel, process}, m_devices{std::move(devices)} {}

GpuMemoryPool::~GpuMemoryPool() {
    for (const auto& [handle, memory] : m_memory) {
        const auto result =
            m_process->GetPageTable().UnlockGpuSharableMemory(memory.address, memory.pinned_size);
        if (result.IsError()) {
            LOG_ERROR(Service, "GPU memory teardown could not release handle={:#x}: {:#x}", handle,
                      result.raw);
        }
        m_devices->ReleaseMemory(memory.device_id);
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
        auto it = m_memory.try_emplace(handle).first;
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

Result GpuMemoryPool::Unregister(u32 handle) {
    const std::scoped_lock lock{m_mutex};
    const auto it = m_memory.find(handle);
    R_UNLESS(it != m_memory.end(), InvalidMemory);
    const auto& memory = it->second;
    R_TRY(m_process->GetPageTable().UnlockGpuSharableMemory(memory.address, memory.pinned_size));
    m_devices->ReleaseMemory(memory.device_id);
    m_memory.erase(it);
    LOG_DEBUG(Service, "GPU memory unregistered: handle={:#x}", handle);
    R_SUCCEED();
}

} // namespace Service::GSV
