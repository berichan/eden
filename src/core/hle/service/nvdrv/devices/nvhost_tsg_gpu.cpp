// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include "common/logging.h"
#include "common/swap.h"
#include "core/hle/service/nvdrv/devices/nvhost_tsg_gpu.h"

namespace Service::Nvidia::Devices {

nvhost_tsg_gpu::nvhost_tsg_gpu(Core::System& system_, NvCore::NativeGpuChannels& channels_)
    : nvdevice{system_}, channels{channels_} {}

nvhost_tsg_gpu::~nvhost_tsg_gpu() = default;

NvResult nvhost_tsg_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output) {
    std::scoped_lock lock{mutex};
    if (!open_fd || *open_fd != fd) {
        return NvResult::InvalidState;
    }
    if (command.group != 'T') {
        return NvResult::NotSupported;
    }
    size_t parameter_size{};
    switch (command.cmd) {
    case 1:
    case 2:
        parameter_size = sizeof(u32_le);
        break;
    case 0xb:
        parameter_size = 6 * sizeof(u32_le);
        break;
    case 7:
    case 0xd:
        parameter_size = 2 * sizeof(u32_le);
        break;
    default:
        LOG_ERROR(Service_NVDRV, "Unsupported TSG ioctl={:08X}", command.raw);
        return NvResult::NotSupported;
    }
    if (command.is_in == 0 || command.is_out == 0) {
        return NvResult::BadParameter;
    }
    if (command.length != parameter_size || input.size() < parameter_size ||
        output.size() < parameter_size) {
        return NvResult::InvalidSize;
    }

    u32_le value;
    std::memcpy(&value, input.data(), sizeof(value));
    NvResult result;
    if (command.cmd == 1 || command.cmd == 2 || command.cmd == 0xb) {
        const auto channel = channels.Get(static_cast<DeviceFD>(static_cast<u32>(value)), owner);
        if (!channel) {
            return NvResult::BadParameter;
        }
        u32_le subcontext{};
        if (command.cmd == 0xb) {
            std::memcpy(&subcontext, input.data() + sizeof(u32_le), sizeof(subcontext));
        }
        result =
            command.cmd == 2 ? channel->UnbindGroup(group) : channel->BindGroup(group, subcontext);
    } else {
        result = command.cmd == 7 ? group->SetInterleave(value) : group->SetL2Promotion(value);
    }
    if (result == NvResult::Success) {
        std::memmove(output.data(), input.data(), parameter_size);
        LOG_DEBUG(Service_NVDRV, "TSG request completed: fd={}, ioctl={:08X}, value={}", fd,
                  command.raw, value);
    }
    return result;
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
        group = std::make_shared<NvCore::NativeGpuSchedulingGroup>(session_id);
        owner = session_id;
        open_fd = fd;
    }
}

void nvhost_tsg_gpu::OnClose(DeviceFD fd) {
    std::scoped_lock lock{mutex};
    if (open_fd && *open_fd == fd) {
        open_fd.reset();
        group.reset();
    }
}

std::optional<nvhost_tsg_gpu::SchedulingParameters> nvhost_tsg_gpu::GetSchedulingParameters()
    const {
    std::scoped_lock lock{mutex};
    return open_fd ? std::optional{group->GetParameters()} : std::nullopt;
}

} // namespace Service::Nvidia::Devices
