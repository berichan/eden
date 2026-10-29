// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/assert.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_qmd.h"
#include "video_core/memory_manager.h"

namespace Tegra::Control {

AmpereGraphicsState::AmpereGraphicsState() {
    registers[0] = ClassId;
    shadow_registers[0] = ClassId;
}

NativeGpuGraphicsContext::NativeGpuGraphicsContext(std::shared_ptr<AmpereGraphicsState> state_,
                                                   std::shared_ptr<MemoryManager> address_space_)
    : state{std::move(state_)}, address_space{std::move(address_space_)} {
    ASSERT(state && address_space);
}

const std::shared_ptr<AmpereGraphicsState>& NativeGpuGraphicsContext::GetState() const {
    return state;
}

const std::shared_ptr<MemoryManager>& NativeGpuGraphicsContext::GetAddressSpace() const {
    return address_space;
}

NativeQmdScheduler::Result NativeGpuGraphicsContext::SignalQmd(GPUVAddr address, u32 argument,
                                                               const QmdBackend& backend) const {
    std::scoped_lock lock{qmd_mutex};
    struct Adapter {
        const NativeGpuGraphicsContext& owner;
        const QmdBackend& backend;
    } adapter{*this, backend};
    const NativeQmdScheduler::Host host{
        &adapter,
        [](void* context, GPUVAddr address, NativeQmdScheduler::Descriptor& descriptor) {
            const auto& self = *static_cast<Adapter*>(context);
            return self.owner.address_space->ReadBlockChecked(
                address, {reinterpret_cast<u8*>(descriptor.data()), sizeof(descriptor)});
        },
        [](void* context, const NativeQmdScheduler::Release& release) {
            const auto& self = *static_cast<Adapter*>(context);
            return !(release.address & 3) &&
                   self.owner.address_space->IsWritableRange(release.address,
                                                             release.SizeBytes()) &&
                   self.backend.validate_release &&
                   self.backend.validate_release(self.backend.context, release);
        },
        [](void* context, const NativeQmdScheduler::Descriptor& descriptor) {
            const auto& backend = static_cast<Adapter*>(context)->backend;
            return backend.dispatch && backend.dispatch(backend.context, descriptor);
        },
        [](void* context, const NativeQmdScheduler::Release& release) {
            const auto& backend = static_cast<Adapter*>(context)->backend;
            return backend.release && backend.release(backend.context, release);
        },
        [](void* context) {
            const auto& backend = static_cast<Adapter*>(context)->backend;
            return backend.should_stop && backend.should_stop(backend.context);
        },
    };
    return qmd_scheduler.Signal(address, argument, host);
}

bool NativeGpuGraphicsContext::HasPendingQmd() const {
    std::scoped_lock lock{qmd_mutex};
    return qmd_scheduler.HasPendingDispatch();
}

bool NativeGpuGraphicsContext::HasQmdFailure() const {
    std::scoped_lock lock{qmd_mutex};
    return qmd_scheduler.HasFailed();
}

bool NativeGpuGraphicsContext::BindZCull(u64 address, u32 mode) {
    if (mode > 3 || (mode == 2 && address == 0)) {
        return false;
    }
    std::scoped_lock lock{state->mutex};
    if (!state->zcull_binding || state->zcull_binding->address == 0) {
        // The driver retains the GPU address without dereferencing it during binding.
        state->zcull_binding = AmpereGraphicsState::ZCullBinding{address, mode};
    }
    return true;
}

std::optional<AmpereGraphicsState::ZCullBinding> NativeGpuGraphicsContext::GetZCullBinding() const {
    std::scoped_lock lock{state->mutex};
    return state->zcull_binding;
}

} // namespace Tegra::Control
