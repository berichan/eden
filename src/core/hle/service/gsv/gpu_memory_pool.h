// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/svc_types.h"
#include "core/hle/result.h"
#include "video_core/host1x/gpu_device_memory_manager.h"

namespace Kernel {
class KProcess;
}

namespace Tegra {
class MemoryManager;
}

namespace Service::GSV {

class GraphicsDeviceTable;
class GpuAddressSpacePool;

class GpuMemoryPool {
public:
    GpuMemoryPool(Kernel::KernelCore& kernel, Kernel::KProcess* process,
                  std::shared_ptr<GraphicsDeviceTable> devices);
    ~GpuMemoryPool();

    Result Register(u32* out_handle, u64 address, u64 last_address, u8 flags, u8 device_id);
    Result Unregister(u32 handle, Kernel::KEvent* completion = nullptr);
    Result RetainReference(u32 handle, u64 address, u64 size, u64 offset);
    Result ReleaseReference(u32 handle);

    struct MapOptions {
        u8 permission;
        u8 cacheability;
        u8 kind;
        u8 compressed_kind;
        u8 raw_texture;
        u8 device_id;
        u8 sync;
    };
    Result Map(u8* out_kind, GpuAddressSpacePool& address_spaces,
               Tegra::MaxwellDeviceMemoryManager& backing, u32 address_space, u64 gpu_address,
               u64 last_address, u32 memory_handle, u64 offset, MapOptions options);
    Result MapSparseMemory(u8* out_kind, GpuAddressSpacePool& address_spaces,
                           Tegra::MaxwellDeviceMemoryManager& backing, u32 address_space,
                           u64 gpu_address, u64 last_address, u32 memory_handle, u64 offset,
                           MapOptions options);
    Result Unmap(u32 address_space, u64 gpu_address, u64 last_address, u64* out_unmapped = nullptr);
    Result UnmapSparseMemory(u32 address_space, u64 gpu_address, u64 last_address,
                             u64* out_unmapped = nullptr);
    Result CreateSparseRegion(GpuAddressSpacePool& address_spaces, u32 address_space,
                              u64 gpu_address, u64 last_address, u8 page_bits, u8 alloc_mode,
                              u8 device_id);
    Result DestroySparseRegion(u32 address_space, u64 gpu_address, u64 last_address);

private:
    Result MapImpl(u8* out_kind, GpuAddressSpacePool& address_spaces,
                   Tegra::MaxwellDeviceMemoryManager& backing, u32 address_space, u64 gpu_address,
                   u64 last_address, u32 memory_handle, u64 offset, MapOptions options,
                   bool sparse);
    Result UnmapImpl(u32 address_space, u64 gpu_address, u64 last_address, u64* out_unmapped,
                     bool sparse);
    struct Memory {
        explicit Memory(Kernel::KernelCore& kernel) : release_event{kernel} {}
        u64 address{};
        u64 pinned_size{};
        u8 device_id{};
        DAddr device_address{};
        size_t mapping_count{};
        size_t reference_count{};
        bool release_pending{};
        Kernel::KScopedAutoObject<Kernel::KEvent> release_event;
        std::vector<Kernel::Svc::lp64::PhysicalMemoryInfo> chunks;
    };

    struct Mapping {
        std::shared_ptr<Tegra::MemoryManager> manager;
        u64 size{};
        u32 memory_handle{};
    };

    struct SparseRegion {
        std::shared_ptr<Tegra::MemoryManager> manager;
        u64 size;
        u8 page_bits;
        u8 alloc_mode;
        u8 device_id;
    };
    using SparseRegions = std::map<std::pair<u32, u64>, SparseRegion>;
    SparseRegions::iterator FindSparseRegion(u32 address_space, u64 address);

    Result ReleaseMemory(std::map<u32, Memory>::iterator it, Kernel::KEvent* completion = nullptr);
    void RemoveMapping(std::map<std::pair<u32, u64>, Mapping>::iterator it,
                       bool restore_sparse = true);

    Kernel::KernelCore& m_kernel;
    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    std::shared_ptr<GraphicsDeviceTable> m_devices;
    std::mutex m_mutex;
    std::map<u32, Memory> m_memory;
    std::map<std::pair<u32, u64>, Mapping> m_mappings;
    SparseRegions m_sparse_regions;
    Tegra::MaxwellDeviceMemoryManager* m_backing{};
    std::optional<Core::Asid> m_asid;
    u32 m_next_handle{};
};

} // namespace Service::GSV
