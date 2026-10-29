// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Control {

// Ampere B's state methods precede the MME invocation window at byte offset 0x3800.
// Method layout: https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clc797.h
struct AmpereGraphicsState {
    static constexpr u32 ClassId = 0xc797;
    static constexpr size_t MethodCount = 0x3800 / sizeof(u32);

    AmpereGraphicsState();

    struct ZCullBinding {
        u64 address;
        u32 mode;
    };

    std::mutex mutex;
    std::array<u32, MethodCount> registers{};
    std::array<u32, MethodCount> shadow_registers{};
    std::optional<ZCullBinding> zcull_binding;
};

// A scheduling group shares graphics state while each address space retains its own binding.
// These are HLE state objects; they do not expose hardware context images to the guest.
class NativeGpuGraphicsContext {
public:
    NativeGpuGraphicsContext(std::shared_ptr<AmpereGraphicsState> state,
                             std::shared_ptr<MemoryManager> address_space);

    const std::shared_ptr<AmpereGraphicsState>& GetState() const;
    const std::shared_ptr<MemoryManager>& GetAddressSpace() const;
    bool BindZCull(u64 address, u32 mode);
    std::optional<AmpereGraphicsState::ZCullBinding> GetZCullBinding() const;

private:
    const std::shared_ptr<AmpereGraphicsState> state;
    const std::shared_ptr<MemoryManager> address_space;
};

} // namespace Tegra::Control
