// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Kernel {
class KClientSession;
class KProcess;
} // namespace Kernel

namespace Service::GSV {

class GpuSession final : public ServiceFramework<GpuSession> {
public:
    explicit GpuSession(Core::System& system, Kernel::KProcess* process);
    ~GpuSession() override;

    Result CreateContext();
    Result InitializeContext();

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

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    KernelHelpers::ServiceContext m_service_context;
    std::array<Kernel::KEvent*, 8> m_events{};
    Kernel::KScopedAutoObject<Kernel::KClientSession> m_worker_client;
    Phase m_phase{Phase::SessionCreated};
};

} // namespace Service::GSV
