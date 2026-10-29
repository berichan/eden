// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/service/gsv/native_memory_descriptions.h"
#include "core/hle/service/service.h"

namespace Kernel {
class KProcess;
class KTransferMemory;
} // namespace Kernel

namespace Service::Nvidia {
class Module;
class NVDRV;
} // namespace Service::Nvidia

namespace Service::GSV {

class SyncptSession;
class GpuSession;
class DisplaySession;
class GpuAddressSpacePool;
class NativeImagePool;

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
    void BootNativeMemoryManagement(HLERequestContext& ctx);
    void CreateNativeMemoryDescription(HLERequestContext& ctx);
    void DestroyNativeMemoryDescription(HLERequestContext& ctx);
    void BindNativeMemoryDescription(HLERequestContext& ctx);
    void CreateGraphicsFrameContext(HLERequestContext& ctx);
    void InitializeGraphicsFrameContext(HLERequestContext& ctx);
    void CreateGpuSessionWithRemoteMemory(HLERequestContext& ctx);
    void CreateGpuContext(HLERequestContext& ctx);
    void InitializeGpuContext(HLERequestContext& ctx);
    void ExistsAddressSpace(HLERequestContext& ctx);
    void CalculateNativeImageSize(HLERequestContext& ctx);
    void CreateNativeImage(HLERequestContext& ctx);
    void DestroyNativeImage(HLERequestContext& ctx);
    void CreateDisplayContext(HLERequestContext& ctx);
    void InitializeDisplayContext(HLERequestContext& ctx);
    void RegisterNativeImageAlias(HLERequestContext& ctx);
    void UnregisterNativeImageAlias(HLERequestContext& ctx);
    void CreateSyncptContext(HLERequestContext& ctx);
    void InitializeSyncptContext(HLERequestContext& ctx);
    void CreateNvDrv2UserSessionWithRemoteMemory(HLERequestContext& ctx);

    Kernel::KScopedAutoObject<Kernel::KProcess> m_process;
    Kernel::KScopedAutoObject<Kernel::KTransferMemory> m_transfer_memory;
    const u64 m_aruid;
    WorkMemoryMode m_work_memory_mode{WorkMemoryMode::Uninitialized};
    u64 m_work_memory_size{};
    bool m_process_bound{};
    bool m_context_created{};
    std::shared_ptr<NativeMemoryDescriptions> m_memory_descriptions;
    std::shared_ptr<NativeImagePool> m_images;
    std::shared_ptr<GpuAddressSpacePool> m_address_spaces;
    std::shared_ptr<Nvidia::Module> m_nvdrv;
    std::weak_ptr<Nvidia::NVDRV> m_nvdrv_user;
    std::shared_ptr<SyncptSession> m_syncpt;
    std::shared_ptr<GpuSession> m_gpu_session;
    std::shared_ptr<DisplaySession> m_display_session;
    bool m_display_initialized{};
    bool m_graphics_frame_created{};
    bool m_graphics_frame_initialized{};
};

} // namespace Service::GSV
