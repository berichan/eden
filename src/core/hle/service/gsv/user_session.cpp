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
           words <= IPC::COMMAND_BUFFER_LENGTH - offset && size <= words * sizeof(u32);
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
      m_transfer_memory{system.Kernel()}, m_aruid{aruid} {
    static const FunctionInfo functions[] = {
        {1, &UserSession::BootWithTransferMemory, "BootWithTransferMemory"},
        {4, &UserSession::BootWithRemoteMemory, "BootWithRemoteMemory"},
        {11, &UserSession::SetCurrentProcessHandle, "SetCurrentProcessHandle"},
        {101, &UserSession::CreateContext, "CreateContext"},
        {201, &UserSession::CreateSyncptContext, "CreateSyncptContext"},
        {202, &UserSession::InitializeSyncptContext, "InitializeSyncptContext"},
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

void UserSession::CreateSyncptContext(HLERequestContext& ctx) {
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
            m_nvdrv = std::make_shared<Nvidia::Module>(system);
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
