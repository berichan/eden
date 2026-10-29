// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/assert.h"
#include "video_core/control/native_compute.h"

namespace Tegra::Control {
namespace {

bool IsStateMethod(u32 method) {
    // Scheduling policy, memory windows and ISA version. Execution and cache
    // operations are handled separately instead of becoming silent register writes.
    switch (method) {
    case 0x7a: // SET_SM_SCG_CONTROL
    case 0x9c: // SET_SCG_CONTROL
    case 0xa8: // SET_SHADER_SHARED_MEMORY_WINDOW_A
    case 0xa9:
    case 0xaa: // SCG_HYSTERESIS_CONTROL
    case 0xac: // SET_CWD_SLOT_COUNT
    case 0xad: // SEND_PCAS_A
    case 0xb3: // SET_SKED_CACHE_CONTROL
    case 0xb9: // SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A
    case 0xba:
    case 0xbb:
    case 0xc4:  // SET_SPA_VERSION
    case 0x1e4: // SET_SHADER_LOCAL_MEMORY_A
    case 0x1e5:
    case 0x1ec: // SET_SHADER_LOCAL_MEMORY_WINDOW_A
    case 0x1ed:
    case 0x365: // SET_SHADER_CACHE_CONTROL
    case 0x379: // SET_SM_TIMEOUT_INTERVAL
        return true;
    default:
        return (method >= 0x368 && method < 0x379) || // scheduling parameters
               (method >= 0xd00 && method < 0xe00);   // MME shadow scratch
    }
}

} // namespace

NativeCompute::NativeCompute(std::shared_ptr<const NativeGpuGraphicsContext> context_)
    : context{std::move(context_)}, macros{context ? context->GetState() : nullptr} {
    ASSERT(context);
}

bool NativeCompute::Fail(Error failure) {
    if (error == Error::None) {
        error = failure;
    }
    return false;
}

bool NativeCompute::CallMethod(u32 method, u32 value, bool last, const Host& host) {
    if (error != Error::None) {
        return false;
    }
    struct Adapter {
        NativeCompute& owner;
        const Host& host;
    } adapter{*this, host};
    const NativeMacroProcessor::Host macro_host{
        &adapter,
        [](void* opaque, u32 method, u32& value) {
            const auto& self = *static_cast<Adapter*>(opaque);
            return self.owner.ReadMethod(method, value);
        },
        [](void* opaque, u32 method, u32 value) {
            auto& self = *static_cast<Adapter*>(opaque);
            return self.owner.WriteMethod(method, value, self.host);
        },
        [](void* opaque) {
            const auto& host = static_cast<Adapter*>(opaque)->host;
            return host.barrier && host.barrier(host.qmd.context);
        },
        [](void* opaque) {
            const auto& qmd = static_cast<Adapter*>(opaque)->host.qmd;
            return qmd.should_stop && qmd.should_stop(qmd.context);
        },
    };
    const auto result = macros.CallMethod(method, value, last, macro_host);
    if (result == NativeMacroProcessor::Status::Failed) {
        return Fail(Error::MacroFailed);
    }
    if (result == NativeMacroProcessor::Status::Accepted) {
        if (method >= 0x45 && method <= 0x48) {
            registers[method] = value;
        }
        return true;
    }
    return WriteMethod(method, value, host);
}

bool NativeCompute::WriteMethod(u32 method, u32 value, const Host& host) {
    if (error != Error::None) {
        return false;
    }
    if (host.qmd.should_stop && host.qmd.should_stop(host.qmd.context)) {
        return Fail(Error::Cancelled);
    }
    if (method >= registers.size()) {
        return Fail(Error::InvalidMethod);
    }
    switch (method) {
    case 0x40: // NO_OPERATION
        return true;
    case 0x44: // WAIT_FOR_IDLE
        if (!host.barrier || !host.barrier(host.qmd.context)) {
            return Fail(Error::UnsupportedMethod);
        }
        break;
    case 0x49: // SET_MME_SHADOW_RAM_CONTROL
        // Replay requires class-specific shadow filtering, which is not implemented yet.
        if (value > 2) {
            return Fail(Error::UnsupportedMethod);
        }
        break;
    case 0xb0: // SEND_SIGNALING_PCAS2_B
        qmd_result = context->SignalQmd(GPUVAddr{registers[0xad]} << 8, value, host.qmd);
        if (!qmd_result.Succeeded()) {
            return Fail(Error::QmdFailed);
        }
        break;
    default:
        if (!IsStateMethod(method)) {
            return Fail(Error::UnsupportedMethod);
        }
        break;
    }
    registers[method] = value;
    return true;
}

u32 NativeCompute::ReadRegister(u32 method) const {
    return method < registers.size() ? registers[method] : 0;
}

bool NativeCompute::ReadMethod(u32 method, u32& value) const {
    if (!IsStateMethod(method) && method != 0x40 && method != 0x44 &&
        !(method >= 0x45 && method <= 0x49) && method != 0xb0) {
        return false;
    }
    value = registers[method];
    return true;
}

NativeCompute::Error NativeCompute::GetError() const {
    return error;
}

bool NativeCompute::HasPendingMacro() const {
    return macros.HasPendingInvocation();
}

const NativeQmdScheduler::Result& NativeCompute::GetQmdResult() const {
    return qmd_result;
}

} // namespace Tegra::Control
