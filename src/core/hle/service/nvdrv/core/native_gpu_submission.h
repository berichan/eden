// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stop_token>
#include <vector>
#include "core/hle/service/nvdrv/core/native_gpu_channel.h"
#include "video_core/control/native_command_stream.h"

namespace Core {
class System;
}

namespace Service::Nvidia::NvCore {

class NativeGpuSubmission {
public:
    // Complete retires this FIFO stream and drains issued work. Dependency-gated QMDs
    // remain in the retained graphics context until a later scheduling signal.
    enum class State { Reserved, Queued, Running, Complete, Failed, Cancelled };
    ~NativeGpuSubmission();
    bool MarkQueued();
    void Execute(Core::System& system, std::stop_token stop_token);
    void Cancel();
    State GetState() const;
    Tegra::Control::NativeCommandStream::Result GetResult() const;
    const std::vector<u64>& GetEntries() const;

private:
    friend class NativeGpuChannel;
    NativeGpuSubmission(std::shared_ptr<NativeGpuChannel> channel, std::vector<u64> entries,
                        NvFence fence);
    void Release();
    void Finish(State state, Tegra::Control::NativeCommandStream::Result result);

    const std::shared_ptr<NativeGpuChannel> channel;
    const std::shared_ptr<const NativeGpuChannel::CommandRing> ring;
    const std::shared_ptr<const Tegra::Control::NativeGpuGraphicsContext> graphics;
    const std::vector<u64> entries;
    const NvFence fence;
    const std::optional<u32> timeout;
    u64 sequence{};
    std::atomic<State> state{State::Reserved};
    std::atomic<bool> released{};
    mutable std::mutex result_mutex;
    Tegra::Control::NativeCommandStream::Result result;
};

} // namespace Service::Nvidia::NvCore
