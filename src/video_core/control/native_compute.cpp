// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/control/native_compute.h"
#include "video_core/memory_manager.h"

namespace Tegra::Control {
namespace {

bool IsStateMethod(u32 method) {
    // Scheduling policy, memory windows and ISA version. Execution and cache
    // operations are handled separately instead of becoming silent register writes.
    switch (method) {
    case 0x7a: // SET_SM_SCG_CONTROL
    case 0x7b: // SET_MME_SWITCH_STATE
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
        return (method >= 0x60 && method <= 0x6b) ||  // inline-DMA parameters
               (method >= 0x368 && method < 0x379) || // scheduling parameters
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
        const auto& execution = macros.GetExecutionResult();
        LOG_ERROR(HW_GPU,
                  "Native compute macro failed: method={:#x}, adapter_error={}, macro_error={}, "
                  "execution_error={}, pc={:#x}, qmd_error={}, qmd_address={:#x}",
                  method, static_cast<u32>(error), static_cast<u32>(macros.GetError()),
                  static_cast<u32>(execution.error), execution.program_counter,
                  static_cast<u32>(qmd_result.error), qmd_result.address);
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
    case 0x7b: { // SET_MME_SWITCH_STATE
        const u32 save = (value >> 4) & 0xff;
        const u32 restore = (value >> 12) & 0xff;
        if ((value & ~0xffff1U) || ((value & 1) && (save >= AmpereGraphicsState::MacroCount ||
                                                    restore >= AmpereGraphicsState::MacroCount))) {
            return Fail(Error::UnsupportedMethod);
        }
        // MME runs without mid-invocation preemption and its state remains host-owned.
        // This configures future switch callbacks; it does not invoke either macro.
        break;
    }
    case 0x91:    // INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, ALL lines
    case 0x4a2:   // INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, ALL lines
    case 0x509:   // INVALIDATE_SAMPLER_CACHE_NO_WFI, ALL lines
    case 0x5a6: { // INVALIDATE_SHADER_CACHES_NO_WFI
        if ((method == 0x5a6 ? (value & ~0x1011U) != 0 : value != 0) || inline_upload.IsPending()) {
            return Fail(Error::UnsupportedMethod);
        }
        NativeCacheInvalidation action;
        if (method != 0x5a6) {
            action.texture_data = method == 0x4a2;
            action.texture_headers = method == 0x91;
            action.samplers = method == 0x509;
        } else {
            action.instructions = (value & 1) != 0;
            action.global_data = (value & 0x10) != 0;
            action.constants = (value & 0x1000) != 0;
        }
        if (!host.invalidate_caches || !host.invalidate_caches(host.qmd.context, action)) {
            return Fail(Error::CacheFailed);
        }
        if (host.qmd.should_stop && host.qmd.should_stop(host.qmd.context)) {
            return Fail(Error::Cancelled);
        }
        break;
    }
    case 0x6c:   // LAUNCH_DMA
    case 0x6d: { // LOAD_INLINE_DATA
        struct UploadContext {
            NativeCompute& owner;
            const Host& host;
        } upload_context{*this, host};
        const NativeInlineUpload::Host upload_host{
            &upload_context,
            [](void* opaque, GPUVAddr address, size_t size) {
                const auto& owner = static_cast<UploadContext*>(opaque)->owner;
                return owner.context->GetAddressSpace()->IsWritableRange(address, size);
            },
            [](void* opaque) {
                const auto& bindings = static_cast<UploadContext*>(opaque)->host;
                return bindings.barrier && bindings.barrier(bindings.qmd.context);
            },
            [](void* opaque, GPUVAddr address, std::span<const u8> bytes) {
                const auto& owner = static_cast<UploadContext*>(opaque)->owner;
                return owner.context->GetAddressSpace()->WriteBlockChecked(address, bytes);
            },
            [](void* opaque) {
                const auto& bindings = static_cast<UploadContext*>(opaque)->host.qmd;
                return bindings.should_stop && bindings.should_stop(bindings.context);
            },
        };
        // C7C0 has a wider upper-address field. The shared collector rejects values
        // outside the native address space's supported 40 bits without truncation.
        const auto result =
            method == 0x6c
                ? inline_upload.Launch({registers[0x60], registers[0x61], registers[0x62],
                                        registers[0x63], registers[0x64]},
                                       value, upload_host)
                : inline_upload.PushWord(value, upload_host);
        if (result != NativeInlineUpload::Error::None) {
            LOG_ERROR(HW_GPU,
                      "Native compute inline DMA failed: method={:#x}, error={}, bytes={:#x}, "
                      "lines={:#x}",
                      method, static_cast<u32>(result), registers[0x60], registers[0x61]);
            return Fail(Error::UploadFailed);
        }
        break;
    }
    case 0xb0: // SEND_SIGNALING_PCAS2_B
        if (inline_upload.IsPending()) {
            return Fail(Error::UploadFailed);
        }
        qmd_result = context->SignalQmd(GPUVAddr{registers[0xad]} << 8, value, host.qmd);
        if (!qmd_result.Succeeded()) {
            return Fail(Error::QmdFailed);
        }
        break;
    default:
        if (!IsStateMethod(method)) {
            LOG_ERROR(HW_GPU, "Unsupported native compute output method={:#x}, value={:#x}", method,
                      value);
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

bool NativeCompute::HasPendingInlineUpload() const {
    return inline_upload.IsPending();
}

const NativeQmdScheduler::Result& NativeCompute::GetQmdResult() const {
    return qmd_result;
}

} // namespace Tegra::Control
