// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <stop_token>
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "video_core/control/native_atomic_sequence.h"

namespace Tegra {
class GPU;
namespace Control {
struct ChannelState;
}
} // namespace Tegra
namespace VideoCore {
class RasterizerInterface;
}

namespace Service::Nvidia::NvCore {

// Internal guarded dispatcher shared by the normal GPU-worker entry point and
// headless routing regressions. No renderer ownership is transferred here.
struct NativeGpuChannel::Backend {
    std::shared_ptr<Tegra::Control::ChannelState> channel;
    std::array<u32, 8> classes{};
    std::array<u32, 0x40> puller{};
    Tegra::Control::NativeAtomicSequence atomic_sequence;

    bool PendingFrontendWork(const Tegra::Control::NativeGpuGraphicsContext& context) const;
    bool PendingFrontend(const Tegra::Control::NativeGpuGraphicsContext& context) const;
    bool Call(Core::System& system, const Tegra::Control::NativeGpuGraphicsContext& context,
              u32 subchannel, u32 method, u32 value, bool last, std::stop_token stop_token);
    bool CallWithGpu(Core::System& system, Tegra::GPU& gpu,
                     VideoCore::RasterizerInterface& rasterizer,
                     const Tegra::Control::NativeGpuGraphicsContext& context, u32 subchannel,
                     u32 method, u32 value, bool last, std::stop_token stop_token);
};

} // namespace Service::Nvidia::NvCore
