// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/control/native_atomic_sequence.h"

namespace Tegra::Control {

bool NativeAtomicSequence::Begin(u32 subchannel, u32 value) {
    // C797/C7C0 FE_ATOMIC_SEQUENCE_BEGIN/END have opaque 32-bit operands.
    // Support the paired 0/1 scopes, without inferring nesting or cross-job semantics.
    if (scope || subchannel >= 8 || value > 1) {
        return false;
    }
    scope = Scope{subchannel, value};
    return true;
}

bool NativeAtomicSequence::End(u32 subchannel, u32 value) {
    if (!scope || scope->subchannel != subchannel || scope->value != value) {
        return false;
    }
    scope.reset();
    return true;
}

bool NativeAtomicSequence::AllowsSubchannel(u32 subchannel) const {
    return subchannel < 8 && (!scope || scope->subchannel == subchannel);
}

bool NativeAtomicSequence::IsActive() const {
    return scope.has_value();
}

} // namespace Tegra::Control
