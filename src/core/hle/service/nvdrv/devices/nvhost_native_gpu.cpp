// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <new>
#include "common/logging.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "core/hle/service/nvdrv/core/native_gpu_error_notifier.h"
#include "core/hle/service/nvdrv/core/native_gpu_submission.h"
#include "core/hle/service/nvdrv/devices/nvhost_native_gpu.h"
#include "video_core/gpu.h"

namespace Service::Nvidia::Devices {

nvhost_native_gpu::nvhost_native_gpu(Core::System& system_, NvCore::NativeGpuChannels& channels_)
    : nvdevice{system_}, channels{channels_} {}

nvhost_native_gpu::~nvhost_native_gpu() {
    if (open_fd) {
        const auto channel = channels.Get(*open_fd, owner);
        channels.Close(*open_fd, owner);
        if (channel && system.IsPoweredOn()) {
            system.GPU().RetireNativeGPUChannel(channel);
        }
    }
}

NvResult nvhost_native_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                   std::span<u8> output) {
    std::scoped_lock lock{mutex};
    if (!open_fd || *open_fd != fd) {
        return NvResult::InvalidState;
    }
    if (command.group == 'G' && (command.cmd == 0x14 || command.cmd == 0x15)) {
        const bool set_data = command.cmd == 0x14;
        if (command.is_in != static_cast<u32>(set_data) ||
            command.is_out != static_cast<u32>(!set_data)) {
            return NvResult::BadParameter;
        }
        if (command.length != sizeof(u64_le) ||
            (set_data ? input.size() : output.size()) < sizeof(u64_le)) {
            return NvResult::InvalidSize;
        }
        if (set_data) {
            u64_le data;
            std::memcpy(&data, input.data(), sizeof(data));
            return channels.SetClientData(fd, owner, data);
        }
        u64 data;
        const auto result = channels.GetClientData(&data, fd, owner);
        if (result == NvResult::Success) {
            const u64_le wire{data};
            std::memcpy(output.data(), &wire, sizeof(wire));
        }
        return result;
    }
    if (command.group != 'H' ||
        (command.cmd != 0x20 && command.cmd != 0x1a && command.cmd != 0x1e && command.cmd != 9 &&
         command.cmd != 0xc && command.cmd != 0x17 && command.cmd != 3 && command.cmd != 0xb)) {
        LOG_ERROR(Service_NVDRV, "Unsupported native channel ioctl={:08X}", command.raw);
        return NvResult::NotSupported;
    }
    const bool allocate_ring = command.cmd == 0x1a;
    const bool input_only = command.cmd == 0x20 || command.cmd == 3;
    if (command.is_in == 0 || command.is_out != static_cast<u32>(!input_only)) {
        return NvResult::BadParameter;
    }
    if (command.cmd == 0xb) {
        constexpr size_t RequestSize = 16;
        if (command.length != RequestSize || input.size() < RequestSize ||
            output.size() < RequestSize) {
            return NvResult::InvalidSize;
        }
        u64_le address;
        u32_le mode;
        std::memcpy(&address, input.data(), sizeof(address));
        std::memcpy(&mode, input.data() + sizeof(address), sizeof(mode));
        const auto result = channels.BindZCull(fd, owner, address, mode);
        if (result == NvResult::Success) {
            std::memmove(output.data(), input.data(), RequestSize);
        }
        LOG_DEBUG(Service_NVDRV, "Native Z-cull binding: fd={}, gpu={:#x}, mode={}, result={}", fd,
                  address, mode, result);
        return result;
    }
    if (command.cmd == 0xc || command.cmd == 0x17) {
        const bool configure = command.cmd == 0xc;
        const size_t request_size = configure ? 24 : 16;
        if (command.length != request_size || input.size() < request_size ||
            output.size() < request_size) {
            return NvResult::InvalidSize;
        }
        const auto channel = channels.Get(fd, owner);
        if (!channel || !channel->IsOpen()) {
            return NvResult::BadParameter;
        }
        const auto& notifier = channel->GetErrorNotifier();
        if (configure) {
            u32_le enable;
            std::memcpy(&enable, input.data() + 16, sizeof(enable));
            notifier->Configure(enable != 0);
            // Native notification records live in the channel, not a guest nvmap allocation.
            std::memmove(output.data(), input.data(), request_size);
        } else {
            const auto notification = notifier->Read();
            const u64_le timestamp{notification.timestamp};
            const u32_le information{notification.information};
            const std::array<u16_le, 2> details{notification.info16, notification.status};
            std::memcpy(output.data(), &timestamp, sizeof(timestamp));
            std::memcpy(output.data() + 8, &information, sizeof(information));
            std::memcpy(output.data() + 12, details.data(), sizeof(details));
        }
        return NvResult::Success;
    }
    if (command.cmd == 9) {
        constexpr size_t RequestSize = 16;
        if (command.length != RequestSize || input.size() < RequestSize ||
            output.size() < RequestSize) {
            return NvResult::InvalidSize;
        }
        std::array<u32_le, 2> params;
        std::memcpy(params.data(), input.data(), sizeof(params));
        const auto result = channels.AllocateObjectContext(fd, owner, params[0], params[1]);
        if (result == NvResult::Success) {
            // The native driver preserves the opaque obj_id field in this ABI.
            std::memmove(output.data(), input.data(), RequestSize);
            LOG_DEBUG(Service_NVDRV, "Native graphics context allocated: fd={}, class={:#x}", fd,
                      params[0]);
        }
        return result;
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
    if (command.cmd == 3) {
        u32_le timeout;
        std::memcpy(&timeout, input.data(), sizeof(timeout));
        return channels.SetTimeout(fd, owner, timeout);
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
    if (!open_fd || *open_fd != fd) {
        return NvResult::InvalidState;
    }
    if (command.group != 'H' || command.cmd != 0x1b) {
        return NvResult::NotSupported;
    }
    constexpr size_t RequestSize = 24;
    if (!command.is_in || !command.is_out) {
        return NvResult::BadParameter;
    }
    if (command.length != RequestSize || input.size() < RequestSize ||
        output.size() < RequestSize) {
        return NvResult::InvalidSize;
    }
    std::array<u32_le, 6> request;
    std::memcpy(request.data(), input.data(), RequestSize);
    const u32 count = request[2];
    if (count > inline_input.size() / sizeof(u64)) {
        return NvResult::InvalidSize;
    }
    if (!count || count > Tegra::Control::NativeCommandStream::MaxEntries) {
        return NvResult::BadParameter;
    }
    try {
        std::vector<u64> entries(count);
        for (u32 index = 0; index < count; ++index) {
            u64_le wire;
            std::memcpy(&wire, inline_input.data() + index * sizeof(wire), sizeof(wire));
            entries[index] = wire;
        }
        std::shared_ptr<NvCore::NativeGpuSubmission> job;
        NvFence fence;
        const auto result =
            channels.ReserveSubmission(&job, &fence, fd, owner, entries, request[3], request[5]);
        if (result != NvResult::Success) {
            return result;
        }
        if (!system.IsPoweredOn()) {
            return NvResult::NotInitialized;
        }
        // The device mutex serializes reservation and queue publication for this channel.
        if (!system.GPU().PushNativeGPUJob(job)) {
            return NvResult::Busy;
        }
        request[3] = 0;
        request[4] = static_cast<u32>(fence.id);
        request[5] = fence.value;
        std::memcpy(output.data(), request.data(), RequestSize);
        LOG_DEBUG(Service_NVDRV,
                  "Native GPU job queued: fd={}, entries={}, syncpoint={}, counter={}", fd, count,
                  fence.id, fence.value);
        return NvResult::Success;
    } catch (const std::bad_alloc&) {
        return NvResult::InsufficientMemory;
    }
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
        const auto channel = channels.Get(fd, owner);
        channels.Close(fd, owner);
        if (channel && system.IsPoweredOn()) {
            system.GPU().RetireNativeGPUChannel(channel);
        }
        open_fd.reset();
    }
}

Kernel::KEvent* nvhost_native_gpu::QueryEvent(u32 event_id) {
    std::scoped_lock lock{mutex};
    if (!open_fd || event_id != 7) {
        return nullptr;
    }
    const auto channel = channels.Get(*open_fd, owner);
    if (!channel || !channel->IsOpen()) {
        return nullptr;
    }
    try {
        return channel->GetErrorNotifier()->GetEvent(system);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

} // namespace Service::Nvidia::Devices
