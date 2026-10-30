// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include "common/common_types.h"
#include "core/hle/result.h"

namespace Tegra::Host1x {
class SyncpointManager;
}

namespace Service::GSV {

// Event ownership stays with the session. Shutdown synchronizes with callbacks
// before the session closes its events; counter notifications never own the session.
class NativeSyncpointWaitPool {
public:
    static constexpr size_t Capacity = 64;
    static constexpr Result InvalidEvent{0x100f3};
    struct Host {
        void* context;
        void (*clear)(void*, u32);
        void (*signal)(void*, u32);
    };

    NativeSyncpointWaitPool(Tegra::Host1x::SyncpointManager& manager, Host host);
    ~NativeSyncpointWaitPool();
    NativeSyncpointWaitPool(const NativeSyncpointWaitPool&) = delete;
    NativeSyncpointWaitPool& operator=(const NativeSyncpointWaitPool&) = delete;
    void Enable(u32 slot);
    Result Arm(u32 slot, u32 counter, u64 target, u8 mode);
    Result Cancel(u32 slot);

private:
    struct Slot;
    Tegra::Host1x::SyncpointManager& m_manager;
    std::array<std::shared_ptr<Slot>, Capacity> m_slots;
};

} // namespace Service::GSV
