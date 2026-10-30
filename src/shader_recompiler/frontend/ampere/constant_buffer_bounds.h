// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include "common/common_types.h"

namespace Shader::IR {
struct Program;
}

namespace Shader::Ampere {

// Bounds describe frozen native bindings. Dynamic/OOB scalar and pair reads
// require 64-byte aligned bounds. Pairs check natural alignment at execution;
// invalid pair addresses return zero without accessing an invalid host packet.
// For 64-byte aligned bindings, proved address ranges reduce
// info.constant_buffer_used_sizes to complete host packets. Declared binding
// bounds remain unchanged; unknown ranges retain the full binding. Unsupported
// loads throw before IR or metadata is changed.
void GuardConstantBufferLoads(IR::Program& program, std::span<const u32> bound_sizes);

} // namespace Shader::Ampere
