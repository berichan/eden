// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Tegra::Host1x {
class SyncpointManager;
}

namespace Service::GSV {

class SyncptSession final : public ServiceFramework<SyncptSession> {
public:
    // An explicitly supplied counter manager must outlive this session.
    explicit SyncptSession(Core::System& system,
                           Tegra::Host1x::SyncpointManager* syncpoints = nullptr);
    ~SyncptSession() override;

private:
    void Ping(HLERequestContext& ctx);
    void GetMaxEventCount(HLERequestContext& ctx);
    void SetEventCount(HLERequestContext& ctx);
    void CreateEvent(HLERequestContext& ctx);
    void QueryCounter(HLERequestContext& ctx);

    // The application quick context has 63 client slots and one additional internal slot.
    static constexpr u32 MaxEventCount = 63;
    KernelHelpers::ServiceContext m_service_context;
    std::array<Kernel::KEvent*, MaxEventCount + 1> m_events{};
    u32 m_event_count{};
    Tegra::Host1x::SyncpointManager* m_syncpoints;
};

} // namespace Service::GSV
