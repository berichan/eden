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

class GpuSession final : public ServiceFramework<GpuSession> {
public:
    explicit GpuSession(Core::System& system, Kernel::KProcess* process);
    ~GpuSession() override;

    Result CreateContext();
    Result InitializeContext();
    Result ExistsAddressSpace(u32 handle);

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
    KernelHelpers::ServiceContext m_service_context;
    std::array<Kernel::KEvent*, 8> m_events{};
    Kernel::KScopedAutoObject<Kernel::KClientSession> m_worker_client;
    std::shared_ptr<GraphicsDeviceTable> m_graphics_devices;
    std::shared_ptr<GpuAddressSpacePool> m_address_spaces;
    Phase m_phase{Phase::SessionCreated};
};

} // namespace Service::GSV
