// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include "common/scope_exit.h"

#include "video_core/control/channel_state.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"

namespace VideoCommon {

template <class P>
ChannelSetupCaches<P>::~ChannelSetupCaches() = default;

template <class P>
void ChannelSetupCaches<P>::CreateChannel(struct Tegra::Control::ChannelState& channel) {
    std::unique_lock<std::mutex> lk(config_mutex);
    ASSERT(channel_map.find(channel.bind_id) == channel_map.end() && channel.bind_id >= 0);
    const bool reused = !free_channel_ids.empty();
    const size_t new_id = reused ? free_channel_ids.front() : channel_storage.size();
    if (reused) {
        channel_storage[new_id].emplace(channel);
    } else {
        channel_storage.emplace_back(std::in_place, channel);
    }
    const size_t as_id = channel.memory_manager->GetID();
    bool mapped{}, active{}, as_held{}, committed{};
    SCOPE_EXIT {
        if (!committed) {
            if (as_held) {
                auto& ref = address_spaces.at(as_id);
                if (--ref.ref_count == 0) {
                    ref.gpu_memory = nullptr;
                    OnGPUASUnregister(as_id);
                }
            }
            if (active) {
                active_channel_ids.pop_back();
            }
            if (mapped) {
                channel_map.erase(channel.bind_id);
            }
            channel_storage[new_id].reset();
            if (!reused) {
                channel_storage.pop_back();
            }
        }
    };
    channel_map.emplace(channel.bind_id, new_id);
    mapped = true;
    active_channel_ids.push_back(new_id);
    active = true;
    auto as_it =
        address_spaces.try_emplace(as_id, AddressSpaceRef{0, address_spaces.size(), nullptr}).first;
    auto& ref = as_it->second;
    const bool retired = ref.ref_count == 0;
    ++ref.ref_count;
    ref.gpu_memory = channel.memory_manager;
    as_held = true;
    if (retired) {
        OnGPUASRegister(as_id);
    }
    if (reused) {
        free_channel_ids.pop_front();
    }
    committed = true;
    if (current_channel_id != UNSET_CHANNEL) {
        channel_state = &*channel_storage[current_channel_id];
    }
}

/// Bind a channel for execution.
template <class P>
void ChannelSetupCaches<P>::BindToChannel(s32 id) {
    std::unique_lock<std::mutex> lk(config_mutex);
    auto it = channel_map.find(id);
    ASSERT(it != channel_map.end() && id >= 0);
    current_channel_id = it->second;
    channel_state = &*channel_storage[current_channel_id];
    maxwell3d = &channel_state->maxwell3d;
    kepler_compute = &channel_state->kepler_compute;
    gpu_memory = &channel_state->gpu_memory;
    program_id = channel_state->program_id;
    current_address_space = gpu_memory->GetID();
}

/// Erase channel's channel_state.
template <class P>
void ChannelSetupCaches<P>::EraseChannel(s32 id) {
    std::unique_lock<std::mutex> lk(config_mutex);
    const auto it = channel_map.find(id);
    if (it == channel_map.end()) {
        return;
    }
    const auto this_id = it->second;
    free_channel_ids.push_back(this_id);
    const size_t as_id = channel_storage[this_id]->gpu_memory.GetID();
    const auto as_it = address_spaces.find(as_id);
    if (as_it != address_spaces.end() && --as_it->second.ref_count == 0) {
        as_it->second.gpu_memory = nullptr;
        OnGPUASUnregister(as_id);
    }
    channel_storage[this_id].reset();
    channel_map.erase(it);
    if (this_id == current_channel_id) {
        current_channel_id = UNSET_CHANNEL;
        channel_state = nullptr;
        maxwell3d = nullptr;
        kepler_compute = nullptr;
        gpu_memory = nullptr;
        program_id = 0;
    } else if (current_channel_id != UNSET_CHANNEL) {
        channel_state = &*channel_storage[current_channel_id];
    }
    const auto active = std::find(active_channel_ids.begin(), active_channel_ids.end(), this_id);
    if (active != active_channel_ids.end()) {
        active_channel_ids.erase(active);
    }
}

} // namespace VideoCommon
