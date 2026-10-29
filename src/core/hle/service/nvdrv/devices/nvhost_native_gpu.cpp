// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include "common/logging.h"
#include "common/swap.h"
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "core/hle/service/nvdrv/devices/nvhost_native_gpu.h"

namespace Service::Nvidia::Devices {

nvhost_native_gpu::nvhost_native_gpu(Core::System& system_, NvCore::NativeGpuChannels& channels_)
    : nvdevice{system_}, channels{channels_} {}

nvhost_native_gpu::~nvhost_native_gpu() {
    if (open_fd) {
        channels.Close(*open_fd, owner);
    }
}

NvResult nvhost_native_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                   std::span<u8> output) {
    std::scoped_lock lock{mutex};
    if (!open_fd || *open_fd != fd) {
        return NvResult::InvalidState;
    }
    if (command.group != 'H' || command.cmd != 0x20) {
        LOG_ERROR(Service_NVDRV, "Unsupported native channel ioctl={:08X}", command.raw);
        return NvResult::NotSupported;
    }
    if (command.is_in == 0 || command.is_out != 0) {
        return NvResult::BadParameter;
    }
    if (command.length != sizeof(u32_le) || input.size() < sizeof(u32_le)) {
        return NvResult::InvalidSize;
    }
    u32_le runlist;
    std::memcpy(&runlist, input.data(), sizeof(runlist));
    const auto result = channels.Allocate(fd, owner, runlist);
    if (result == NvResult::Success) {
        LOG_DEBUG(Service_NVDRV, "Native channel allocated: fd={}, requested_runlist={}", fd,
                  runlist);
    }
    return result;
}

NvResult nvhost_native_gpu::Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                   std::span<const u8> inline_input, std::span<u8> output) {
    std::scoped_lock lock{mutex};
    return open_fd && *open_fd == fd ? NvResult::NotSupported : NvResult::InvalidState;
}

NvResult nvhost_native_gpu::Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                   std::span<u8> output, std::span<u8> inline_output) {
    std::scoped_lock lock{mutex};
    return open_fd && *open_fd == fd ? NvResult::NotSupported : NvResult::InvalidState;
}

void nvhost_native_gpu::OnOpen(NvCore::SessionId session_id, DeviceFD fd) {
    std::scoped_lock lock{mutex};
    if (!open_fd) {
        open_fd = fd;
        owner = session_id;
    }
}

void nvhost_native_gpu::OnClose(DeviceFD fd) {
    std::scoped_lock lock{mutex};
    if (open_fd && *open_fd == fd) {
        channels.Close(fd, owner);
        open_fd.reset();
    }
}

} // namespace Service::Nvidia::Devices
