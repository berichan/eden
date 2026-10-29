// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include "core/hle/service/nvdrv/devices/nvdevice.h"

namespace Service::Nvidia::Devices {

class nvhost_tsg_gpu final : public nvdevice {
public:
    struct SchedulingParameters {
        u32 interleave_level{};
        u32 l2_promotion{};
    };

    explicit nvhost_tsg_gpu(Core::System& system);
    ~nvhost_tsg_gpu() override;

    NvResult Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                    std::span<u8> output) override;
    NvResult Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                    std::span<const u8> inline_input, std::span<u8> output) override;
    NvResult Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output,
                    std::span<u8> inline_output) override;

    void OnOpen(NvCore::SessionId session_id, DeviceFD fd) override;
    void OnClose(DeviceFD fd) override;

    std::optional<SchedulingParameters> GetSchedulingParameters() const;

private:
    mutable std::mutex mutex;
    std::optional<DeviceFD> open_fd;
    SchedulingParameters parameters;
};

} // namespace Service::Nvidia::Devices
