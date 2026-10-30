// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/assert.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_macro_processor.h"

namespace Tegra::Control {
namespace {

bool IsUpload(u32 method) {
    return method >= 0x45 && method <= 0x48;
}

} // namespace

struct NativeMacroProcessor::PrivateProgram {
    explicit PrivateProgram(const AmpereGraphicsState& state)
        : instructions{state.macro_engine.CopyProgram()}, starts{state.macro_starts},
          bind_pointer{state.macro_bind_pointer} {}
    TuringMacroEngine::Program instructions;
    std::array<std::optional<u32>, AmpereGraphicsState::MacroCount> starts;
    u32 bind_pointer;
};

NativeMacroProcessor::NativeMacroProcessor(std::shared_ptr<AmpereGraphicsState> state_,
                                           ProgramBank bank_)
    : state{std::move(state_)}, bank{bank_} {
    ASSERT(state);
}

NativeMacroProcessor::~NativeMacroProcessor() = default;

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
    if (method == 0x15d) { // SET_MME_DATA_FIFO_CONFIG, 0/4/8/12/16 KiB
        if (value > 4)
            return fail(Error::InvalidDmaConfiguration);
        fifo_capacity_words = value * 1024;
        return Status::Accepted;
    }
    if (method < MacroMethodStart && !IsUpload(method)) {
        return Status::Unhandled;
    }
    std::scoped_lock lock{state->mutex};
    if (bank == ProgramBank::Compute && !private_program && IsUpload(method)) {
        // Compute can invoke the graphics-uploaded library before it installs its
        // own program. Queue variants pack that library at different addresses;
        // an explicit compute upload must not overwrite graphics or another queue.
        private_program = std::make_unique<PrivateProgram>(*state);
    }
    auto& starts = private_program ? private_program->starts : state->macro_starts;
    auto& bind_pointer =
        private_program ? private_program->bind_pointer : state->macro_bind_pointer;
    if (method == 0x45) {
        const bool uploaded =
            private_program
                ? TuringMacroEngine::SetInstructionPointer(private_program->instructions, value)
                : state->macro_engine.SetInstructionPointer(value);
        return uploaded ? Status::Accepted : fail(Error::InvalidUpload);
    }
    if (method == 0x46) {
        const bool uploaded =
            private_program
                ? TuringMacroEngine::UploadInstruction(private_program->instructions, value)
                : state->macro_engine.UploadInstruction(value);
        return uploaded ? Status::Accepted : fail(Error::InvalidUpload);
    }
    if (method == 0x47) {
        if (value >= starts.size()) {
            return fail(Error::InvalidBinding);
        }
        bind_pointer = value;
        return Status::Accepted;
    }
    if (method == 0x48) {
        if (bind_pointer >= starts.size() || value >= TuringMacroEngine::InstructionCount) {
            return fail(Error::InvalidBinding);
        }
        starts[bind_pointer++] = value;
        return Status::Accepted;
    }
    const u32 slot = (method - MacroMethodStart) / 2;
    if (slot >= starts.size() || !starts[slot] || (!invocation && (method & 1))) {
        return fail(Error::InvalidInvocation);
    }
    if (!invocation) {
        invocation = method;
        invocation_start = *starts[slot];
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
        u32 fifo_capacity_words;
        size_t cursor{};
        std::vector<u32> dma_parameters;
        GPUVAddr dma_address{};
        u32 dma_count{};
        bool dma_active{};
    } invocation_host{host, parameters, fifo_capacity_words};
    const TuringMacroEngine::Host bindings{
        &invocation_host,
        [](void* context, u32& parameter) {
            auto& self = *static_cast<InvocationHost*>(context);
            const auto& source = self.dma_active ? self.dma_parameters : self.parameters;
            if (self.cursor == source.size()) {
                return false;
            }
            parameter = source[self.cursor++];
            return true;
        },
        [](void* context, u16 address, u32& result) {
            const auto& host = static_cast<InvocationHost*>(context)->host;
            return !(address & 3) && address / 4 < MacroMethodStart && host.read_method &&
                   host.read_method(host.context, address / 4, result);
        },
        [](void* context, u16 address, u32 data) {
            auto& self = *static_cast<InvocationHost*>(context);
            const auto& host = self.host;
            if (address == 0x158 * 4) { // MME_DMA_READ_FIFOED, length in words
                const auto& source = self.dma_active ? self.dma_parameters : self.parameters;
                u32 upper{}, lower{};
                // Do not overwrite an outstanding transfer or unread DMA words.
                // The switch from method parameters to DMA occurs at the barrier.
                if (self.dma_count || (self.dma_active && self.cursor != source.size()) ||
                    data == 0 || data > self.fifo_capacity_words || !host.read_memory ||
                    !host.read_method || !host.read_method(host.context, 0x154, upper) ||
                    !host.read_method(host.context, 0x155, lower) || upper > 0xff || (lower & 3))
                    return false;
                const GPUVAddr gpu_address = (GPUVAddr{upper} << 32) | lower;
                if (u64{data} * 4 > (u64{1} << 40) - gpu_address)
                    return false;
                self.dma_address = gpu_address;
                self.dma_count = data;
                return true;
            }
            // Until all DMA input is resolved and consumed, no emitted method
            // may publish an action or completion ahead of a later FIFO failure.
            if (self.dma_count || (self.dma_active && self.cursor != self.dma_parameters.size()))
                return false;
            // Mutating instruction/start RAM during an invocation is not supported.
            return !(address & 3) && address / 4 < MacroMethodStart && !IsUpload(address / 4) &&
                   host.write_method && host.write_method(host.context, address / 4, data);
        },
        [](void* context) {
            auto& self = *static_cast<InvocationHost*>(context);
            const auto& host = self.host;
            const auto& source = self.dma_active ? self.dma_parameters : self.parameters;
            if (self.dma_count && self.cursor != source.size())
                return false;
            if (!host.barrier || !host.barrier(host.context) ||
                (host.should_stop && host.should_stop(host.context)))
                return false;
            if (self.dma_count) {
                self.dma_parameters.resize(self.dma_count);
                if (!host.read_memory(host.context, self.dma_address, self.dma_parameters) ||
                    (host.should_stop && host.should_stop(host.context)))
                    return false;
                self.dma_count = 0;
                self.cursor = 0;
                self.dma_active = true;
            }
            return true;
        },
        [](void* context) {
            const auto& host = static_cast<InvocationHost*>(context)->host;
            return host.should_stop && host.should_stop(host.context);
        },
    };
    execution_result = private_program
                           ? state->macro_engine.Execute(private_program->instructions, start,
                                                         bindings, instruction_limit)
                           : state->macro_engine.Execute(start, bindings, instruction_limit);
    if (!execution_result.Succeeded()) {
        // A binding can refer to a different upload than an earlier capture. Keep
        // fault diagnostics bounded to the failing instruction and its predecessors.
        const auto fault = execution_result.program_counter;
        const auto first = fault > 3 ? fault - 3 : 0;
        for (u32 pc = first; pc <= fault && pc < TuringMacroEngine::InstructionCount; ++pc) {
            TuringMacroEngine::EncodedInstruction words;
            const bool uploaded =
                private_program
                    ? TuringMacroEngine::ReadInstruction(private_program->instructions, pc, words)
                    : state->macro_engine.ReadInstruction(pc, words);
            if (!uploaded)
                continue;
            const auto inst = TuringMacroEngine::Decode(words);
            LOG_DEBUG(
                HW_GPU,
                "Native macro fault instruction: pc={:#x}, end={}, predicate={}/{}, "
                "alu0={}/{}/{}/{}/{:#x}, alu1={}/{}/{}/{}/{:#x}, output0={}/{}, output1={}/{}",
                pc, inst.end_next, inst.predicate_mode, inst.predicate,
                static_cast<u32>(inst.alu[0].operation), inst.alu[0].destination,
                inst.alu[0].source_a, inst.alu[0].source_b, inst.immediate[0],
                static_cast<u32>(inst.alu[1].operation), inst.alu[1].destination,
                inst.alu[1].source_a, inst.alu[1].source_b, inst.immediate[1],
                inst.output[0].method, inst.output[0].emit, inst.output[1].method,
                inst.output[1].emit);
        }
    }
    if (!execution_result.Succeeded())
        return fail(Error::ExecutionFailed);
    if (invocation_host.dma_count ||
        (invocation_host.dma_active &&
         invocation_host.cursor != invocation_host.dma_parameters.size()))
        return fail(Error::UnfinishedDma);
    return Status::Accepted;
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

std::span<const u32> NativeMacroProcessor::InvocationParameters() const {
    return parameters;
}

} // namespace Tegra::Control
