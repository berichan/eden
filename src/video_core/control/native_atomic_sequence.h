// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include "common/common_types.h"

namespace Tegra::Control {

// Restricted FE scopes for a worker that executes an entire job without channel preemption.
// This does not implement physical GPU preemption, rollback, or a memory-completion barrier.
class NativeAtomicSequence {
public:
    bool Begin(u32 subchannel, u32 value);
    bool End(u32 subchannel, u32 value);
    bool AllowsSubchannel(u32 subchannel) const;
    bool IsActive() const;

private:
    struct Scope {
        u32 subchannel;
        u32 value;
    };
    std::optional<Scope> scope;
};

} // namespace Tegra::Control
