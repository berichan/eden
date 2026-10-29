// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/user_session.h"
#include "core/hle/service/hle_ipc.h"
#include "core/hle/service/ipc_helpers.h"

namespace {

// An unscheduled dummy caller is enough to decode messages without starting a guest or allocating
// RAM.
struct Caller {
    Core::System system;
    Kernel::KProcess process{system.Kernel()};
    Kernel::KThread thread{system.Kernel()};

    Caller() {
        Kernel::KAutoObject::Create(&process);
        system.Kernel().SetCurrentEmuThread(&thread);
        REQUIRE(Kernel::KThread::InitializeDummyThread(system, &thread, &process).IsSuccess());
    }
    ~Caller() {
        auto& kernel = system.Kernel();
        process.UnregisterThread(kernel, &thread);
        // Release the dummy caller's owner reference; the root is on the stack.
        process.Close(kernel);
        kernel.UnregisterKernelObject(&thread);
        kernel.UnregisterKernelObject(&process);
        kernel.SetCurrentEmuThread(nullptr);
    }
};

template <typename T>
void Put(std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH>& data, size_t& offset, const T& value) {
    std::memcpy(data.data() + offset, &value, sizeof(value));
    offset += sizeof(value) / sizeof(u32);
}

Result Request(Caller& caller, Service::GSV::UserSession& service, u32 command,
               std::span<const u64> arguments) {
    std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> data{};
    size_t offset{};
    IPC::CommandHeader header{};
    header.type.Assign(static_cast<IPC::CommandType>(16 + command));
    header.enable_handle_descriptor.Assign(1);
    header.data_size.Assign(static_cast<u32>(arguments.size() * 2));
    Put(data, offset, header);
    IPC::HandleDescriptorHeader handles{};
    handles.num_handles_to_copy.Assign(1);
    Put(data, offset, handles);
    Put(data, offset, u32{0xffff8001});
    for (const auto argument : arguments) {
        Put(data, offset, argument);
    }
    Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(), nullptr,
                                   &caller.thread};
    REQUIRE(ctx.PopulateFromIncomingCommandBuffer(data.data()).IsSuccess());
    service.InvokeRequestTipc(ctx);
    IPC::CommandHeader output{};
    std::memcpy(&output, ctx.CommandBuffer(), sizeof(output));
    REQUIRE(output.data_size == 1);
    Result result{};
    result.raw = ctx.CommandBuffer()[2];
    return result;
}

} // namespace

TEST_CASE("TIPC arguments follow handle and buffer descriptors", "[core][ipc]") {
    Caller caller;
    for (const bool handles : {false, true}) {
        for (const bool buffers : {false, true}) {
            CAPTURE(handles, buffers);
            std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> data{};
            size_t offset{};
            IPC::CommandHeader header{};
            header.type.Assign(static_cast<IPC::CommandType>(16 + 4));
            header.enable_handle_descriptor.Assign(handles);
            header.num_buf_a_descriptors.Assign(buffers);
            header.data_size.Assign(12);
            Put(data, offset, header);
            if (handles) {
                IPC::HandleDescriptorHeader descriptor{};
                descriptor.num_handles_to_copy.Assign(1);
                descriptor.num_handles_to_move.Assign(1);
                descriptor.send_current_pid.Assign(1);
                Put(data, offset, descriptor);
                Put(data, offset, u64{0x1111222233334444});
                Put(data, offset, u32{0xffff8001});
                Put(data, offset, u32{0x1234});
            }
            if (buffers) {
                Put(data, offset, IPC::BufferDescriptorABW{});
            }
            const auto payload_offset = offset;
            const std::array<u64, 6> arguments{0x80, 0x8000000000000001, 0, 0, 0, 0x12345678};
            Put(data, offset, arguments);
            Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(),
                                           nullptr, &caller.thread};
            REQUIRE(ctx.PopulateFromIncomingCommandBuffer(data.data()).IsSuccess());
            REQUIRE(ctx.GetCommand() == 4);
            REQUIRE(ctx.GetDataPayloadOffset() == payload_offset);
            if (handles) {
                REQUIRE(ctx.GetCopyHandle(0) == 0xffff8001);
                REQUIRE(ctx.GetMoveHandle(0) == 0x1234);
            }
            IPC::RequestParser parser{ctx};
            for (const auto expected : arguments) {
                REQUIRE(parser.Pop<u64>() == expected);
            }
        }
    }
}

TEST_CASE("TIPC replies pack outputs immediately after the four byte result", "[core][ipc]") {
    Caller caller;
    std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> data{};
    IPC::CommandHeader header{};
    header.type.Assign(static_cast<IPC::CommandType>(16));
    std::memcpy(data.data(), &header, sizeof(header));
    Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(), nullptr,
                                   &caller.thread};
    REQUIRE(ctx.PopulateFromIncomingCommandBuffer(data.data()).IsSuccess());
    IPC::ResponseBuilder reply{ctx, 4};
    reply.Push(ResultSuccess);
    reply.Push<u64>(0x1122334455667788);
    IPC::CommandHeader output{};
    std::memcpy(&output, ctx.CommandBuffer(), sizeof(output));
    REQUIRE(output.data_size == 3);
    REQUIRE(reply.GetCurrentOffset() == 5);
    REQUIRE(ctx.CommandBuffer()[2] == 0);
    REQUIRE(ctx.CommandBuffer()[3] == 0x55667788);
    REQUIRE(ctx.CommandBuffer()[4] == 0x11223344);
}

TEST_CASE("GSV remote bootstrap validates the owner and supported memory state",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::UserSession service{caller.system, &caller.process, 0x55};
    std::array<u64, 6> arguments{0x55, 0x8000000000000001, 0, 0, 0, 0};
    Result expected = Kernel::ResultNotImplemented;
    SECTION("Wrong applet owner") {
        arguments[0] = 0x56;
        expected = Kernel::ResultInvalidArgument;
    }
    SECTION("Unknown ABI version") {
        arguments[1] = 0;
    }
    SECTION("Unsupported reserved input") {
        arguments[2] = 1;
    }
    SECTION("Existing remote mappings require an allocator") {
        arguments[5] = 1;
    }
    SECTION("Truncated input") {
        REQUIRE(Request(caller, service, 4, std::span{arguments}.first(5)) ==
                Kernel::ResultInvalidSize);
        return;
    }
    SECTION("Transfer memory requires a transfer-memory handle") {
        const std::array<u64, 2> transfer_arguments{0x55, 0x1000};
        REQUIRE(Request(caller, service, 1, transfer_arguments) == Kernel::ResultInvalidHandle);
        return;
    }
    REQUIRE(Request(caller, service, 4, arguments) == expected);
    arguments = {0x55, 0x8000000000000001, 0, 0, 0, 0};
    REQUIRE(Request(caller, service, 4, arguments).IsSuccess());
}

TEST_CASE("GSV creates a user context only after boot and process binding", "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::UserSession service{caller.system, &caller.process, 0x55};
    const std::array<u64, 6> arguments{0x55, 0x8000000000000001, 0, 0, 0, 0};
    REQUIRE(Request(caller, service, 101, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 11, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 4, arguments).IsSuccess());
    REQUIRE(Request(caller, service, 4, arguments) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 101, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 11, {}).IsSuccess());
    REQUIRE(Request(caller, service, 101, {}).IsSuccess());
    REQUIRE(Request(caller, service, 101, {}) == Kernel::ResultInvalidState);
}
