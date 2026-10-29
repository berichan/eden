// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>

#include "common/alignment.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_transfer_memory.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/display_session.h"
#include "core/hle/service/gsv/gpu_address_space_pool.h"
#include "core/hle/service/gsv/gpu_session.h"
#include "core/hle/service/gsv/syncpt_session.h"
#include "core/hle/service/gsv/user_session.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/nvdrv/nvdrv_interface.h"

namespace Service::GSV {
namespace {

bool HasInputData(HLERequestContext& ctx, size_t size) {
    IPC::CommandHeader header{};
    std::memcpy(&header, ctx.CommandBuffer(), sizeof(header));
    const size_t offset = ctx.GetDataPayloadOffset();
    const size_t words = header.data_size;
    return ctx.IsTipc() && offset <= IPC::COMMAND_BUFFER_LENGTH &&
           words <= IPC::COMMAND_BUFFER_LENGTH - offset && size == words * sizeof(u32);
}

void Reply(HLERequestContext& ctx, Result result) {
    if (result.IsError()) {
        LOG_WARNING(Service, "GSV command {} failed: {:#x}", ctx.GetCommand(), result.raw);
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result);
}

} // namespace

UserSession::UserSession(Core::System& system, Kernel::KProcess* process, u64 aruid)
    : ServiceFramework{system, "GsvUserSession"}, m_process{system.Kernel(), process},
      m_transfer_memory{system.Kernel()}, m_aruid{aruid},
      m_address_spaces{std::make_shared<GpuAddressSpacePool>()} {
    static const FunctionInfo functions[] = {
        {1, &UserSession::BootWithTransferMemory, "BootWithTransferMemory"},
        {4, &UserSession::BootWithRemoteMemory, "BootWithRemoteMemory"},
        {11, &UserSession::SetCurrentProcessHandle, "SetCurrentProcessHandle"},
        {101, &UserSession::CreateContext, "CreateContext"},
        {115, &UserSession::BootNativeMemoryManagement, "BootNativeMemoryManagement"},
        {201, &UserSession::CreateSyncptContext, "CreateSyncptContext"},
        {202, &UserSession::InitializeSyncptContext, "InitializeSyncptContext"},
        {301, &UserSession::CreateDisplayContext, "CreateDisplayContext"},
        {305, &UserSession::InitializeDisplayContext, "InitializeDisplayContext"},
        {401, &UserSession::CreateGpuContext, "CreateGpuContext"},
        {402, &UserSession::InitializeGpuContext, "InitializeGpuContext"},
        {434, &UserSession::ExistsAddressSpace, "ExistsAddressSpace"},
        {497, &UserSession::CreateGpuSessionWithRemoteMemory, "CreateGpuSessionWithRemoteMemory"},
        {1101, &UserSession::CreateGraphicsFrameContext, "CreateGraphicsFrameContext"},
        {1102, &UserSession::InitializeGraphicsFrameContext, "InitializeGraphicsFrameContext"},
        {1402, &UserSession::CreateNvDrv2UserSessionWithRemoteMemory,
         "CreateNvDrv2UserSessionWithRemoteMemory"},
    };
    RegisterHandlersTipc(functions);
}

UserSession::~UserSession() = default;

void UserSession::BootWithTransferMemory(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, 2 * sizeof(u64)), Kernel::ResultInvalidSize);
        R_UNLESS(m_work_memory_mode == WorkMemoryMode::Uninitialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto aruid = rp.Pop<u64>();
        const auto size = rp.Pop<u64>();
        auto memory = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(0));
        R_UNLESS(memory.IsNotNull(), Kernel::ResultInvalidHandle);
        R_UNLESS(aruid == m_aruid && memory->GetOwner() == m_process.GetPointerUnsafe(),
                 Kernel::ResultInvalidArgument);
        R_UNLESS(size != 0 && Common::IsAligned(size, Kernel::PageSize) &&
                     size <= memory->GetSize(),
                 Kernel::ResultInvalidSize);
        m_transfer_memory.SetObject(memory.GetPointerUnsafe());
        m_work_memory_mode = WorkMemoryMode::Transfer;
        m_work_memory_size = size;
        LOG_DEBUG(Service, "GSV work memory attached: aruid={:#x}, size={:#x}", aruid, size);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void UserSession::BootWithRemoteMemory(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, 6 * sizeof(u64)), Kernel::ResultInvalidSize);
        R_UNLESS(m_work_memory_mode == WorkMemoryMode::Uninitialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto aruid = rp.Pop<u64>();
        const auto version = rp.Pop<u64>();
        const auto reserved = rp.PopRaw<std::array<u64, 3>>();
        const auto usage = rp.Pop<u64>();
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(ctx.GetCopyHandle(0));
        R_UNLESS(process.IsNotNull(), Kernel::ResultInvalidHandle);
        R_UNLESS(aruid == m_aruid && process.GetPointerUnsafe() == m_process.GetPointerUnsafe(),
                 Kernel::ResultInvalidArgument);
        R_UNLESS(version == 0x8000000000000001ULL || version == 0x8000000000000005ULL,
                 Kernel::ResultNotImplemented);
        R_UNLESS((reserved == std::array<u64, 3>{}) && usage == 0, Kernel::ResultNotImplemented);
        // There are no remote mappings in the current memory profile. Boot attaches the process;
        // allocation commands must separately account for and map any memory they create.
        m_work_memory_mode = WorkMemoryMode::Remote;
        LOG_DEBUG(Service, "GSV remote context attached: aruid={:#x}, version={:#x}, usage={:#x}",
                  aruid, version, usage);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void UserSession::SetCurrentProcessHandle(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, 0), Kernel::ResultInvalidSize);
        R_UNLESS(m_work_memory_mode != WorkMemoryMode::Uninitialized, Kernel::ResultInvalidState);
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(ctx.GetCopyHandle(0));
        R_UNLESS(process.IsNotNull(), Kernel::ResultInvalidHandle);
        R_UNLESS(process.GetPointerUnsafe() == m_process.GetPointerUnsafe(),
                 Kernel::ResultInvalidArgument);
        m_process_bound = true;
        LOG_DEBUG(Service, "GSV process bound: aruid={:#x}", m_aruid);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void UserSession::CreateContext(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, 0), Kernel::ResultInvalidSize);
        R_UNLESS(m_process_bound && !m_context_created, Kernel::ResultInvalidState);
        m_context_created = true;
        LOG_DEBUG(Service, "GSV user context created: aruid={:#x}, work_memory_size={:#x}", m_aruid,
                  m_work_memory_size);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void UserSession::BootNativeMemoryManagement(HLERequestContext& ctx) {
    u64 modes{};
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, sizeof(u64)), Kernel::ResultInvalidSize);
        R_UNLESS(m_context_created, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto requested_modes = rp.Pop<u64>();
        R_UNLESS((requested_modes & ~u64{2}) == 0, Kernel::ResultNotImplemented);
        // This negotiates the client's block mode. It does not allocate or map a memory region;
        // mode bit 1 uses SVC 0x80 and separate native-memory creation commands for those steps.
        modes = requested_modes;
        LOG_DEBUG(Service, "GSV native memory modes negotiated: {:#x}", modes);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(result);
    rb.Push(modes);
}

void UserSession::CreateGraphicsFrameContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    if (!m_context_created) {
        Reply(ctx, Kernel::ResultInvalidState);
        return;
    }
    m_graphics_frame_created = true;
    Reply(ctx, ResultSuccess);
}

void UserSession::InitializeGraphicsFrameContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    if (!m_graphics_frame_created) {
        Reply(ctx, Kernel::ResultInvalidState);
        return;
    }
    // No frame has been submitted yet. Frame scheduling and mode commands require separate
    // implementations beyond this context's initialization.
    m_graphics_frame_initialized = true;
    LOG_DEBUG(Service, "GSV graphics frame context initialized: aruid={:#x}", m_aruid);
    Reply(ctx, ResultSuccess);
}

void UserSession::CreateGpuSessionWithRemoteMemory(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, sizeof(u64)), Kernel::ResultInvalidSize);
        R_UNLESS(m_context_created && m_work_memory_mode == WorkMemoryMode::Remote,
                 Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto version = rp.Pop<u64>();
        R_UNLESS(version == 2 || version == 3, Kernel::ResultNotImplemented);
        R_UNLESS(!m_gpu_session, Result{0x200f3});
        auto gpu =
            std::make_shared<GpuSession>(system, m_process.GetPointerUnsafe(), m_address_spaces);
        R_TRY(ctx.TryAddCopyInterface(gpu));
        m_gpu_session = std::move(gpu);
        LOG_DEBUG(Service, "Native GPU session created: aruid={:#x}, version={}", m_aruid, version);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void UserSession::CreateGpuContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    Reply(ctx, m_gpu_session ? m_gpu_session->CreateContext() : Kernel::ResultInvalidState);
}

void UserSession::InitializeGpuContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    Reply(ctx, m_gpu_session ? m_gpu_session->InitializeContext() : Kernel::ResultInvalidState);
}

void UserSession::ExistsAddressSpace(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, sizeof(u32)), Kernel::ResultInvalidSize);
        R_UNLESS(m_gpu_session, GpuAddressSpacePool::InvalidAddressSpace);
        IPC::RequestParser rp{ctx};
        const auto handle = rp.Pop<u32>();
        R_TRY(m_gpu_session->ExistsAddressSpace(handle));
        LOG_DEBUG(Service, "GSV native GPU address space verified: handle={:#x}", handle);
        R_SUCCEED();
    }();
    Reply(ctx, result);
}

void UserSession::CreateDisplayContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    if (!m_context_created) {
        Reply(ctx, Kernel::ResultInvalidState);
        return;
    }
    if (!m_display_session) {
        m_display_session = std::make_shared<DisplaySession>(system, m_process.GetPointerUnsafe());
    }
    Reply(ctx, ResultSuccess);
}

void UserSession::InitializeDisplayContext(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, sizeof(u64)), Kernel::ResultInvalidSize);
        R_UNLESS(m_display_session && !m_display_initialized, Kernel::ResultInvalidState);
        IPC::RequestParser rp{ctx};
        const auto version = rp.Pop<u64>();
        R_UNLESS(version == 0x300010000ULL, Kernel::ResultNotImplemented);
        R_TRY(ctx.TryAddCopyInterface(m_display_session));
        m_display_initialized = true;
        LOG_DEBUG(Service, "Native display session initialized: aruid={:#x}, version={:#x}",
                  m_aruid, version);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void UserSession::CreateSyncptContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    if (!m_context_created) {
        Reply(ctx, Kernel::ResultInvalidState);
        return;
    }
    if (!m_syncpt) {
        m_syncpt = std::make_shared<SyncptSession>(system);
    }
    Reply(ctx, ResultSuccess);
}

void UserSession::InitializeSyncptContext(HLERequestContext& ctx) {
    if (!HasInputData(ctx, 0)) {
        Reply(ctx, Kernel::ResultInvalidSize);
        return;
    }
    const auto result = m_syncpt ? ctx.TryAddCopyInterface(m_syncpt) : Kernel::ResultInvalidState;
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

void UserSession::CreateNvDrv2UserSessionWithRemoteMemory(HLERequestContext& ctx) {
    const auto result = [&]() -> Result {
        R_UNLESS(HasInputData(ctx, 0), Kernel::ResultInvalidSize);
        R_UNLESS(m_context_created && m_work_memory_mode == WorkMemoryMode::Remote,
                 Kernel::ResultInvalidState);
        R_UNLESS(m_nvdrv_user.expired(), Result{0x200f3});
        if (!m_nvdrv) {
            // Keep native device state local to this graphics context, including across reopen.
            m_nvdrv = std::make_shared<Nvidia::Module>(system, m_address_spaces);
        }
        auto user = std::make_shared<Nvidia::NVDRV>(system, m_nvdrv, "nvdrv2:user",
                                                    m_process.GetPointerUnsafe());
        R_TRY(ctx.TryAddCopyInterface(user));
        m_nvdrv_user = user;
        LOG_DEBUG(Service, "Native driver session created: aruid={:#x}", m_aruid);
        R_SUCCEED();
    }();
    IPC::ResponseBuilder rb{ctx, 2, result.IsSuccess() ? 1U : 0U};
    rb.Push(result);
}

} // namespace Service::GSV
