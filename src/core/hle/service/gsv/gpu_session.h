// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Kernel {
class KClientSession;
class KProcess;
} // namespace Kernel

namespace Service::GSV {

class GraphicsDeviceTable;
class GpuAddressSpacePool;
class GpuMemoryPool;
class GpuUserEvents;
struct NativeImagePreference;

class GpuSession final : public ServiceFramework<GpuSession> {
public:
    explicit GpuSession(Core::System& system, Kernel::KProcess* process,
                        std::shared_ptr<GpuAddressSpacePool> address_spaces = {});
    ~GpuSession() override;

    Result CreateContext();
    Result InitializeContext();
    Result ExistsAddressSpace(u32 handle);
    Result CalculateNativeImageSize(u64* out_size, const NativeImagePreference& preference,
                                    u32 layers);
    Result GetNativeMemoryPool(std::shared_ptr<GpuMemoryPool>* out);

private:
    enum class Phase {
        SessionCreated,
        ContextCreated,
        Initialized,
    };

    void GetEventIdRange(HLERequestContext& ctx);
    void CreateEvent(HLERequestContext& ctx);
    void RequestKUserWorkerSession(HLERequestContext& ctx);
    void GetKUserWorkerSession(HLERequestContext& ctx);
    void CreateGraphicsDevice(HLERequestContext& ctx);
    void DestroyGraphicsDevice(HLERequestContext& ctx);
    void CreateGpuAddressSpace(HLERequestContext& ctx);
    void GetMaxGpuAddressSpaceId(HLERequestContext& ctx);
    void GetGpuPartitionRange(HLERequestContext& ctx);

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    std::shared_ptr<GpuUserEvents> m_events;
    Kernel::KScopedAutoObject<Kernel::KClientSession> m_worker_client;
    std::shared_ptr<GraphicsDeviceTable> m_graphics_devices;
    std::shared_ptr<GpuAddressSpacePool> m_address_spaces;
    std::shared_ptr<GpuMemoryPool> m_memory;
    Phase m_phase{Phase::SessionCreated};
};

} // namespace Service::GSV
