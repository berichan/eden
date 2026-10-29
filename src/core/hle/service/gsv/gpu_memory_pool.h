// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/kernel/svc_types.h"
#include "core/hle/result.h"

namespace Kernel {
class KProcess;
}

namespace Service::GSV {

class GraphicsDeviceTable;

class GpuMemoryPool {
public:
    GpuMemoryPool(Kernel::KernelCore& kernel, Kernel::KProcess* process,
                  std::shared_ptr<GraphicsDeviceTable> devices);
    ~GpuMemoryPool();

    Result Register(u32* out_handle, u64 address, u64 last_address, u8 flags, u8 device_id);
    Result Unregister(u32 handle);

private:
    struct Memory {
        u64 address{};
        u64 pinned_size{};
        u8 device_id{};
        std::vector<Kernel::Svc::lp64::PhysicalMemoryInfo> chunks;
    };

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    std::shared_ptr<GraphicsDeviceTable> m_devices;
    std::mutex m_mutex;
    std::map<u32, Memory> m_memory;
    u32 m_next_handle{};
};

} // namespace Service::GSV
