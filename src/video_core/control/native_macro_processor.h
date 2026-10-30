// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <optional>
#include <vector>
#include "common/common_types.h"
#include "video_core/turing_macro.h"

namespace Tegra::Control {

struct AmpereGraphicsState;

class NativeMacroProcessor {
public:
    static constexpr u32 MacroMethodStart = 0xe00;
    static constexpr size_t MaxParameters = 0x1fff;

    struct Host {
        void* context{};
        bool (*read_method)(void*, u32 word_method, u32&){};
        bool (*write_method)(void*, u32 word_method, u32){};
        bool (*barrier)(void*){};
        bool (*should_stop)(void*){};
    };
    enum class Status { Unhandled, Accepted, Failed };
    enum class ProgramBank { Graphics, Compute };
    enum class Error {
        None,
        Cancelled,
        InvalidUpload,
        InvalidBinding,
        InvalidInvocation,
        ParameterLimit,
        ExecutionFailed,
    };

    explicit NativeMacroProcessor(std::shared_ptr<AmpereGraphicsState> state,
                                  ProgramBank bank = ProgramBank::Graphics);
    ~NativeMacroProcessor();
    Status CallMethod(u32 method, u32 value, bool last, const Host& host,
                      u64 instruction_limit = 1'000'000);
    Error GetError() const;
    const TuringMacroEngine::Result& GetExecutionResult() const;
    bool HasPendingInvocation() const;

private:
    struct PrivateProgram;
    std::shared_ptr<AmpereGraphicsState> state;
    const ProgramBank bank;
    std::unique_ptr<PrivateProgram> private_program;
    std::optional<u32> invocation;
    u32 invocation_start{};
    std::vector<u32> parameters;
    Error error{};
    TuringMacroEngine::Result execution_result;
};

} // namespace Tegra::Control
