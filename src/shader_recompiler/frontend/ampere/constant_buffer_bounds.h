// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include "common/common_types.h"

namespace Shader::IR {
struct Program;
}

namespace Shader::Ampere {

// Bounds describe frozen native bindings. Dynamic/OOB scalar reads require
// 64-byte aligned bounds; unsupported loads throw before the IR is changed.
void GuardConstantBufferLoads(IR::Program& program, std::span<const u32> bound_sizes);

} // namespace Shader::Ampere
