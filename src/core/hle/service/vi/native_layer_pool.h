// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include "common/common_types.h"
#include "core/hle/result.h"

namespace Service::VI {

class NativeLayerPool {
public:
    static constexpr size_t Capacity = 64;

    Result Create(u64* out_id, u64 owner_process_id) {
        auto slot = std::ranges::find_if(m_layers, [](const auto& layer) { return !layer; });
        R_UNLESS(slot != m_layers.end() && m_next_id != std::numeric_limits<u64>::max(),
                 (Result{ErrorModule::VI, 1}));
        *slot = Layer{m_next_id++, owner_process_id};
        *out_id = (*slot)->id;
        R_SUCCEED();
    }

    Result Destroy(u64 id, u64 owner_process_id) {
        const auto slot = std::ranges::find_if(m_layers, [=](const auto& layer) {
            return layer && layer->id == id && layer->owner_process_id == owner_process_id;
        });
        R_UNLESS(slot != m_layers.end(), (Result{ErrorModule::VI, 7}));
        slot->reset();
        R_SUCCEED();
    }

    Result QueryBufferState(u64* out_state, u64 id, u64 owner_process_id) {
        const auto* layer = Find(id, owner_process_id);
        R_UNLESS(layer != nullptr, (Result{ErrorModule::VI, 7}));
        *out_state = layer->buffer_state;
        R_SUCCEED();
    }

    Result Bind(u64 id, u64 owner_process_id, u64 display_id) {
        auto* layer = Find(id, owner_process_id);
        R_UNLESS(layer != nullptr, (Result{ErrorModule::VI, 7}));
        R_UNLESS(!layer->display_id, (Result{ErrorModule::VI, 1}));
        layer->display_id = display_id;
        R_SUCCEED();
    }

    Result Unbind(u64 id, u64 owner_process_id) {
        auto* layer = Find(id, owner_process_id);
        R_UNLESS(layer != nullptr, (Result{ErrorModule::VI, 7}));
        layer->display_id.reset();
        R_SUCCEED();
    }

    void Clear() {
        m_layers.fill(std::nullopt);
    }

private:
    // Native image storage belongs to subsequent buffer registration, not layer allocation.
    struct Layer {
        u64 id;
        u64 owner_process_id;
        u64 buffer_state{};
        std::optional<u64> display_id{};
    };

    Layer* Find(u64 id, u64 owner_process_id) {
        const auto slot = std::ranges::find_if(m_layers, [=](const auto& layer) {
            return layer && layer->id == id && layer->owner_process_id == owner_process_id;
        });
        return slot != m_layers.end() ? &**slot : nullptr;
    }

    std::array<std::optional<Layer>, Capacity> m_layers{};
    u64 m_next_id{1};
};

} // namespace Service::VI
