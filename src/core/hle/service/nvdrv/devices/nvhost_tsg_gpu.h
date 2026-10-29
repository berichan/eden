// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "core/hle/service/nvdrv/devices/nvdevice.h"

namespace Service::Nvidia::Devices {

class nvhost_tsg_gpu final : public nvdevice {
public:
    using SchedulingParameters = NvCore::NativeGpuSchedulingGroup::Parameters;

    nvhost_tsg_gpu(Core::System& system, NvCore::NativeGpuChannels& channels);
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
    NvCore::NativeGpuChannels& channels;
    std::optional<DeviceFD> open_fd;
    NvCore::SessionId owner{};
    std::shared_ptr<NvCore::NativeGpuSchedulingGroup> group;
};

} // namespace Service::Nvidia::Devices
