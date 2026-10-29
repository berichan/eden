// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/assert.h"
#include "video_core/control/native_gpu_context.h"

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
