// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <optional>
#include <span>
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
        bool (*read_memory)(void*, GPUVAddr, std::span<u32>){};
    };
    enum class Status { Unhandled, Accepted, Failed };
    // Explicit compute uploads fork program and data together; bootstrap calls share graphics.
    enum class ProgramBank { Graphics, Compute };
    enum class Error {
        None,
        Cancelled,
        InvalidUpload,
        InvalidBinding,
        InvalidInvocation,
        ParameterLimit,
        ExecutionFailed,
        UnfinishedDma,
        InvalidDmaConfiguration,
    };

    explicit NativeMacroProcessor(std::shared_ptr<AmpereGraphicsState> state,
                                  ProgramBank bank = ProgramBank::Graphics);
    ~NativeMacroProcessor();
    Status CallMethod(u32 method, u32 value, bool last, const Host& host,
                      u64 instruction_limit = 1'000'000);
    Error GetError() const;
    const TuringMacroEngine::Result& GetExecutionResult() const;
    bool HasPendingInvocation() const;
    // GPU-worker callback view, valid only until the current CallMethod returns.
    std::span<const u32> InvocationParameters() const;

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
    // Bounded software capacity until explicitly configured; not a hardware reset value.
    u32 fifo_capacity_words = 1024;
};

} // namespace Tegra::Control
