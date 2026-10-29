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

} // namespace Tegra::Control
