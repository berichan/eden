// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/service/service.h"

namespace Kernel {
class KProcess;
class KTransferMemory;
} // namespace Kernel

namespace Service::GSV {

class UserSession final : public ServiceFramework<UserSession> {
public:
    explicit UserSession(Core::System& system, Kernel::KProcess* process, u64 aruid);
    ~UserSession() override;

private:
    enum class WorkMemoryMode {
        Uninitialized,
        Transfer,
        Remote,
    };

    void BootWithTransferMemory(HLERequestContext& ctx);
    void BootWithRemoteMemory(HLERequestContext& ctx);
    void SetCurrentProcessHandle(HLERequestContext& ctx);
    void CreateContext(HLERequestContext& ctx);

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    Kernel::KScopedAutoObject<Kernel::KTransferMemory> m_transfer_memory;
    const u64 m_aruid;
    WorkMemoryMode m_work_memory_mode{WorkMemoryMode::Uninitialized};
    u64 m_work_memory_size{};
    bool m_process_bound{};
    bool m_context_created{};
};

} // namespace Service::GSV
