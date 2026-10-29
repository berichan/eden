// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/assert.h"
#include "common/scope_exit.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_macro_processor.h"

namespace Tegra::Control {
namespace {

bool IsUpload(u32 method) {
    return method >= 0x45 && method <= 0x48;
}

} // namespace

NativeMacroProcessor::NativeMacroProcessor(std::shared_ptr<AmpereGraphicsState> state_)
    : state{std::move(state_)} {
    ASSERT(state);
}

NativeMacroProcessor::Status NativeMacroProcessor::CallMethod(u32 method, u32 value, bool last,
                                                              const Host& host,
                                                              u64 instruction_limit) {
    const auto fail = [&](Error failure) {
        error = failure;
        return Status::Failed;
    };
    if (error != Error::None) {
        return Status::Failed;
    }
    if (host.should_stop && host.should_stop(host.context)) {
        return fail(Error::Cancelled);
    }
    if (invocation && method != *invocation + 1) {
        return fail(Error::InvalidInvocation);
    }
    if (method < MacroMethodStart && !IsUpload(method)) {
        return Status::Unhandled;
    }
    std::scoped_lock lock{state->mutex};
    if (method == 0x45) {
        return state->macro_engine.SetInstructionPointer(value) ? Status::Accepted
                                                                : fail(Error::InvalidUpload);
    }
    if (method == 0x46) {
        return state->macro_engine.UploadInstruction(value) ? Status::Accepted
                                                            : fail(Error::InvalidUpload);
    }
    if (method == 0x47) {
        if (value >= state->macro_starts.size()) {
            return fail(Error::InvalidBinding);
        }
        state->macro_bind_pointer = value;
        return Status::Accepted;
    }
    if (method == 0x48) {
        if (state->macro_bind_pointer >= state->macro_starts.size() ||
            value >= TuringMacroEngine::InstructionCount) {
            return fail(Error::InvalidBinding);
        }
        state->macro_starts[state->macro_bind_pointer++] = value;
        return Status::Accepted;
    }
    const u32 slot = (method - MacroMethodStart) / 2;
    if (slot >= state->macro_starts.size() || !state->macro_starts[slot] ||
        (!invocation && (method & 1))) {
        return fail(Error::InvalidInvocation);
    }
    if (!invocation) {
        invocation = method;
        invocation_start = *state->macro_starts[slot];
    }
    if (parameters.size() == MaxParameters) {
        return fail(Error::ParameterLimit);
    }
    parameters.push_back(value);
    if (!last) {
        return Status::Accepted;
    }
    const u32 start = invocation_start;
    invocation.reset();
    SCOPE_EXIT {
        parameters.clear();
    };
    struct InvocationHost {
        const Host& host;
        const std::vector<u32>& parameters;
        size_t cursor{};
    } invocation_host{host, parameters};
    const TuringMacroEngine::Host bindings{
        &invocation_host,
        [](void* context, u32& parameter) {
            auto& self = *static_cast<InvocationHost*>(context);
            if (self.cursor == self.parameters.size()) {
                return false;
            }
            parameter = self.parameters[self.cursor++];
            return true;
        },
        [](void* context, u16 address, u32& result) {
            const auto& host = static_cast<InvocationHost*>(context)->host;
            return !(address & 3) && address / 4 < MacroMethodStart && host.read_method &&
                   host.read_method(host.context, address / 4, result);
        },
        [](void* context, u16 address, u32 data) {
            const auto& host = static_cast<InvocationHost*>(context)->host;
            // Mutating instruction/start RAM during an invocation is not supported.
            return !(address & 3) && address / 4 < MacroMethodStart && !IsUpload(address / 4) &&
                   host.write_method && host.write_method(host.context, address / 4, data);
        },
        [](void* context) {
            const auto& host = static_cast<InvocationHost*>(context)->host;
            return host.barrier && host.barrier(host.context);
        },
        [](void* context) {
            const auto& host = static_cast<InvocationHost*>(context)->host;
            return host.should_stop && host.should_stop(host.context);
        },
    };
    execution_result = state->macro_engine.Execute(start, bindings, instruction_limit);
    return execution_result.Succeeded() ? Status::Accepted : fail(Error::ExecutionFailed);
}

NativeMacroProcessor::Error NativeMacroProcessor::GetError() const {
    return error;
}

const TuringMacroEngine::Result& NativeMacroProcessor::GetExecutionResult() const {
    return execution_result;
}

bool NativeMacroProcessor::HasPendingInvocation() const {
    return invocation.has_value();
}

} // namespace Tegra::Control
