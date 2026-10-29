// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include "common/logging.h"
#include "common/swap.h"
#include "core/hle/service/nvdrv/devices/nvhost_tsg_gpu.h"

namespace Service::Nvidia::Devices {

nvhost_tsg_gpu::nvhost_tsg_gpu(Core::System& system_) : nvdevice{system_} {}

nvhost_tsg_gpu::~nvhost_tsg_gpu() = default;

NvResult nvhost_tsg_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output) {
    std::scoped_lock lock{mutex};
    if (!open_fd || *open_fd != fd) {
        return NvResult::InvalidState;
    }
    if (command.group != 'T' || (command.cmd != 7 && command.cmd != 0xd)) {
        // Opening a scheduling group does not allocate a GPU channel. Channel binding,
        // submission and scheduler operations need their own implementations.
        LOG_ERROR(Service_NVDRV, "Unsupported TSG ioctl={:08X}", command.raw);
        return NvResult::NotSupported;
    }
    if (command.is_in == 0 || command.is_out == 0) {
        return NvResult::BadParameter;
    }
    constexpr size_t ParameterSize = 2 * sizeof(u32_le);
    if (command.length != ParameterSize || input.size() < ParameterSize ||
        output.size() < ParameterSize) {
        return NvResult::InvalidSize;
    }

    u32_le value;
    std::memcpy(&value, input.data(), sizeof(value));
    if (command.cmd == 7) {
        if (value > 2) {
            return NvResult::BadParameter;
        }
        parameters.interleave_level = value;
    } else {
        if (value != 1 && value != 2 && value != 4) {
            return NvResult::BadParameter;
        }
        parameters.l2_promotion = value;
    }
    // The second word is reserved and is preserved, not interpreted as another setting.
    std::memmove(output.data(), input.data(), ParameterSize);
    LOG_DEBUG(Service_NVDRV, "TSG configuration: fd={}, interleave={}, l2_promotion={}", fd,
              parameters.interleave_level, parameters.l2_promotion);
    return NvResult::Success;
}

NvResult nvhost_tsg_gpu::Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<const u8> inline_input, std::span<u8> output) {
    std::scoped_lock lock{mutex};
    return open_fd && *open_fd == fd ? NvResult::NotSupported : NvResult::InvalidState;
}

NvResult nvhost_tsg_gpu::Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output, std::span<u8> inline_output) {
    std::scoped_lock lock{mutex};
    return open_fd && *open_fd == fd ? NvResult::NotSupported : NvResult::InvalidState;
}

void nvhost_tsg_gpu::OnOpen(NvCore::SessionId session_id, DeviceFD fd) {
    std::scoped_lock lock{mutex};
    if (!open_fd) {
        open_fd = fd;
        parameters = {};
    }
}

void nvhost_tsg_gpu::OnClose(DeviceFD fd) {
    std::scoped_lock lock{mutex};
    if (open_fd && *open_fd == fd) {
        open_fd.reset();
        parameters = {};
    }
}

std::optional<nvhost_tsg_gpu::SchedulingParameters> nvhost_tsg_gpu::GetSchedulingParameters()
    const {
    std::scoped_lock lock{mutex};
    return open_fd ? std::optional{parameters} : std::nullopt;
}

} // namespace Service::Nvidia::Devices
