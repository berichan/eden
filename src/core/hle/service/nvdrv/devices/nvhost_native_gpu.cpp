// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
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
    if (command.group != 'H' ||
        (command.cmd != 0x20 && command.cmd != 0x1a && command.cmd != 0x1e)) {
        LOG_ERROR(Service_NVDRV, "Unsupported native channel ioctl={:08X}", command.raw);
        return NvResult::NotSupported;
    }
    const bool allocate_ring = command.cmd == 0x1a;
    if (command.is_in == 0 || command.is_out != static_cast<u32>(command.cmd != 0x20)) {
        return NvResult::BadParameter;
    }
    if (command.cmd == 0x1e) {
        if (command.length != 16 || input.size() < 16 || output.size() < 16) {
            return NvResult::InvalidSize;
        }
        u64 address;
        NvFence fence;
        const auto result = channels.QuerySyncpoint(&address, &fence, fd, owner);
        if (result == NvResult::Success) {
            const u64_le wire_address{address};
            const std::array<u32_le, 2> wire_fence{static_cast<u32>(fence.id), fence.value};
            std::memcpy(output.data(), &wire_address, sizeof(wire_address));
            std::memcpy(output.data() + sizeof(wire_address), wire_fence.data(),
                        sizeof(wire_fence));
            LOG_DEBUG(
                Service_NVDRV,
                "Native channel syncpoint queried: fd={}, gpu={:#x}, syncpoint={}, counter={}", fd,
                address, fence.id, fence.value);
        }
        return result;
    }
    if (allocate_ring) {
        constexpr size_t RequestSize = 0x20;
        if (command.length != RequestSize || input.size() < RequestSize ||
            output.size() < RequestSize) {
            return NvResult::InvalidSize;
        }
        std::array<u32_le, 8> params;
        std::memcpy(params.data(), input.data(), RequestSize);
        NvFence fence;
        const auto result =
            channels.AllocateCommandRing(&fence, fd, owner, params[0], params[1], params[2]);
        if (result == NvResult::Success) {
            params[3] = static_cast<u32>(fence.id);
            params[4] = fence.value;
            std::memcpy(output.data(), params.data(), RequestSize);
            LOG_DEBUG(Service_NVDRV,
                      "Native command ring allocated: fd={}, entries={}, jobs={}, flags={:#x}, "
                      "syncpoint={}, counter={}",
                      fd, params[0], params[1], params[2], fence.id, fence.value);
        }
        return result;
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
