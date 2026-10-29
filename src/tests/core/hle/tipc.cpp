// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <utility>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/gsv/display_event_queue.h"
#include "core/hle/service/gsv/display_session.h"
#include "core/hle/service/gsv/gpu_session.h"
#include "core/hle/service/gsv/syncpt_session.h"
#include "core/hle/service/gsv/user_session.h"
#include "core/hle/service/hle_ipc.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/vi/native_layer_pool.h"

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
               std::span<const u64> arguments, u64* out_value = nullptr) {
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
    REQUIRE(output.data_size == (out_value ? 3U : 1U));
    Result result{};
    result.raw = ctx.CommandBuffer()[2];
    if (out_value) {
        std::memcpy(out_value, ctx.CommandBuffer() + 3, sizeof(*out_value));
    }
    return result;
}

std::pair<Result, u64> RequestWords(Caller& caller, Service::ServiceFrameworkBase& service,
                                    u32 command, std::span<const u32> arguments,
                                    u32 reply_words = 1) {
    std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> data{};
    size_t offset{};
    IPC::CommandHeader header{};
    header.type.Assign(static_cast<IPC::CommandType>(16 + command));
    header.data_size.Assign(static_cast<u32>(arguments.size()));
    Put(data, offset, header);
    for (const auto argument : arguments) {
        Put(data, offset, argument);
    }
    Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(), nullptr,
                                   &caller.thread};
    REQUIRE(ctx.PopulateFromIncomingCommandBuffer(data.data()).IsSuccess());
    service.InvokeRequestTipc(ctx);
    IPC::CommandHeader output{};
    std::memcpy(&output, ctx.CommandBuffer(), sizeof(output));
    REQUIRE(output.enable_handle_descriptor == 0);
    REQUIRE(output.data_size == reply_words);
    const u64 value = u64{ctx.CommandBuffer()[3]} | (u64{ctx.CommandBuffer()[4]} << 32);
    return {Result{ctx.CommandBuffer()[2]}, value};
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
    REQUIRE(Request(caller, service, 1402, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 201, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 202, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 1101, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 1102, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 401, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 402, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 301, {}) == Kernel::ResultInvalidState);
    const std::array<u64, 1> display_version{0x300010000};
    REQUIRE(Request(caller, service, 305, display_version) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 11, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 4, arguments).IsSuccess());
    REQUIRE(Request(caller, service, 4, arguments) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 101, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 1402, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 11, {}).IsSuccess());
    REQUIRE(Request(caller, service, 101, {}).IsSuccess());
    REQUIRE(Request(caller, service, 101, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 1102, {}) == Kernel::ResultInvalidState);
    REQUIRE(Request(caller, service, 1101, {}).IsSuccess());
    REQUIRE(Request(caller, service, 1102, {}).IsSuccess());
    REQUIRE(Request(caller, service, 301, {}).IsSuccess());
    REQUIRE(Request(caller, service, 305, {}) == Kernel::ResultInvalidSize);
    const std::array<u64, 1> invalid_display_version{0};
    REQUIRE(Request(caller, service, 305, invalid_display_version) == Kernel::ResultNotImplemented);
    const std::array<u64, 1> unsupported_gpu_version{4};
    REQUIRE(Request(caller, service, 497, unsupported_gpu_version) == Kernel::ResultNotImplemented);
    REQUIRE(Request(caller, service, 201, {}).IsSuccess());
    REQUIRE(Request(caller, service, 201, {}).IsSuccess());
}

TEST_CASE("GSV syncpoint ping echoes the complete value and rejects truncated data",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::SyncptSession service{caller.system};
    for (const u32 words : {0U, 1U, 2U, 3U}) {
        CAPTURE(words);
        std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> data{};
        size_t offset{};
        IPC::CommandHeader header{};
        header.type.Assign(static_cast<IPC::CommandType>(16));
        header.data_size.Assign(words);
        Put(data, offset, header);
        Put(data, offset, u64{0xfedcba9876543210});
        Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(), nullptr,
                                       &caller.thread};
        REQUIRE(ctx.PopulateFromIncomingCommandBuffer(data.data()).IsSuccess());
        service.InvokeRequestTipc(ctx);
        IPC::CommandHeader output{};
        std::memcpy(&output, ctx.CommandBuffer(), sizeof(output));
        if (words == 2) {
            REQUIRE(output.data_size == 3);
            REQUIRE(ctx.CommandBuffer()[2] == ResultSuccess.raw);
            REQUIRE(ctx.CommandBuffer()[3] == 0x76543210);
            REQUIRE(ctx.CommandBuffer()[4] == 0xfedcba98);
        } else {
            REQUIRE(output.data_size == 1);
            REQUIRE(ctx.CommandBuffer()[2] == Kernel::ResultInvalidSize.raw);
        }
    }
}

TEST_CASE("GSV native memory boot negotiates modes without allocating regions",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::UserSession service{caller.system, &caller.process, 0x55};
    const std::array<u64, 1> supported{2};
    u64 output{0xffff};
    REQUIRE(Request(caller, service, 115, supported, &output) == Kernel::ResultInvalidState);
    REQUIRE(output == 0);
    const std::array<u64, 6> boot{0x55, 0x8000000000000001, 0, 0, 0, 0};
    REQUIRE(Request(caller, service, 4, boot).IsSuccess());
    REQUIRE(Request(caller, service, 11, {}).IsSuccess());
    REQUIRE(Request(caller, service, 101, {}).IsSuccess());
    REQUIRE(Request(caller, service, 115, {}, &output) == Kernel::ResultInvalidSize);
    const std::array<u64, 1> unsupported{8};
    REQUIRE(Request(caller, service, 115, unsupported, &output) == Kernel::ResultNotImplemented);
    REQUIRE(output == 0);
    REQUIRE(Request(caller, service, 115, supported, &output).IsSuccess());
    REQUIRE(output == 2);
    const std::array<u64, 1> none{0};
    REQUIRE(Request(caller, service, 115, none, &output).IsSuccess());
    REQUIRE(output == 0);
}

TEST_CASE("GSV syncpoint event pools validate initialization and index limits",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::SyncptSession service{caller.system};
    const auto [result, count] = RequestWords(caller, service, 1, {}, 2);
    REQUIRE(result.IsSuccess());
    REQUIRE(count == 63);
    const std::array<u32, 1> zero{0};
    REQUIRE(RequestWords(caller, service, 22, zero).first.IsError());
    REQUIRE(RequestWords(caller, service, 21, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 1> too_many{64};
    REQUIRE(RequestWords(caller, service, 21, too_many).first == Result{0x80f3});
    SECTION("Full application event pool") {
        const std::array<u32, 1> max_events{63};
        REQUIRE(RequestWords(caller, service, 21, max_events).first.IsSuccess());
        REQUIRE(RequestWords(caller, service, 22, too_many).first == Result{0x100f3});
        const std::array<u32, 1> overflow{0xffffffff};
        REQUIRE(RequestWords(caller, service, 22, overflow).first == Result{0x100f3});
    }
    SECTION("Empty client pool retains only its internal slot") {
        REQUIRE(RequestWords(caller, service, 21, zero).first.IsSuccess());
        const std::array<u32, 1> first_client{1};
        REQUIRE(RequestWords(caller, service, 22, first_client).first == Result{0x100f3});
    }
    REQUIRE(RequestWords(caller, service, 21, zero).first == Result{0x200f3});
}

TEST_CASE("Unknown service commands return protocol errors unless auto stubs are enabled",
          "[core][ipc]") {
    const auto saved_auto_stub = Settings::values.use_auto_stub.GetValue();
    const auto saved_asserts = Settings::values.use_debug_asserts.GetValue();
    const auto saved_reporting = Settings::values.reporting_services.GetValue();
    SCOPE_EXIT {
        Settings::values.use_auto_stub.SetValue(saved_auto_stub);
        Settings::values.use_debug_asserts.SetValue(saved_asserts);
        Settings::values.reporting_services.SetValue(saved_reporting);
    };
    Settings::values.use_debug_asserts.SetValue(false);
    Settings::values.reporting_services.SetValue(false);
    Caller caller;
    Service::GSV::SyncptSession service{caller.system};
    for (const bool auto_stub : {false, true}) {
        Settings::values.use_auto_stub.SetValue(auto_stub);
        for (const bool tipc : {false, true}) {
            CAPTURE(auto_stub, tipc);
            std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> data{};
            size_t offset{};
            IPC::CommandHeader header{};
            header.type.Assign(tipc ? static_cast<IPC::CommandType>(16 + 0x7fff)
                                    : IPC::CommandType::Request);
            header.data_size.Assign(tipc ? 2 : 8);
            Put(data, offset, header);
            if (!tipc) {
                offset = 4;
                IPC::DataPayloadHeader payload{};
                payload.magic = Common::MakeMagic('S', 'F', 'C', 'I');
                Put(data, offset, payload);
                Put(data, offset, u64{0x7fff});
            }
            Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(),
                                           nullptr, &caller.thread};
            REQUIRE(ctx.PopulateFromIncomingCommandBuffer(data.data()).IsSuccess());
            if (tipc) {
                service.InvokeRequestTipc(ctx);
            } else {
                service.InvokeRequest(ctx);
            }
            const Result expected = auto_stub ? ResultSuccess
                                    : tipc    ? Result{ErrorModule::TIPC, 10}
                                              : Result{ErrorModule::CMIF, 221};
            REQUIRE(ctx.CommandBuffer()[tipc ? 2 : 6] == expected.raw);
            if (tipc) {
                IPC::CommandHeader reply{};
                std::memcpy(&reply, ctx.CommandBuffer(), sizeof(reply));
                REQUIRE(reply.data_size == 1);
            } else {
                REQUIRE(ctx.CommandBuffer()[4] == Common::MakeMagic('S', 'F', 'C', 'O'));
                REQUIRE(ctx.CommandBuffer()[7] == 0);
            }
        }
    }
}

TEST_CASE("Native GPU control sessions validate phases and event bounds", "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::GpuSession service{caller.system, &caller.process};
    REQUIRE(service.InitializeContext() == Kernel::ResultInvalidState);
    REQUIRE(RequestWords(caller, service, 11, {}).first == Kernel::ResultInvalidState);
    REQUIRE(RequestWords(caller, service, 12, {}).first == Kernel::ResultInvalidSize);
    REQUIRE(service.CreateContext().IsSuccess());
    REQUIRE(service.CreateContext() == Kernel::ResultInvalidState);
    REQUIRE(service.InitializeContext().IsSuccess());
    REQUIRE(service.InitializeContext() == Kernel::ResultInvalidState);
    const auto [range_result, range] = RequestWords(caller, service, 11, {}, 3);
    REQUIRE(range_result.IsSuccess());
    REQUIRE(range == 0x700000000ULL);
    const std::array<u32, 1> invalid_id{8};
    REQUIRE(RequestWords(caller, service, 12, invalid_id).first == Kernel::ResultInvalidArgument);
    const std::array<u32, 1> overflowing_id{0xffffffff};
    REQUIRE(RequestWords(caller, service, 12, overflowing_id).first ==
            Kernel::ResultInvalidArgument);
    const std::array<u32, 1> missing_event{0};
    REQUIRE(RequestWords(caller, service, 21, missing_event).first ==
            Kernel::ResultInvalidArgument);
    REQUIRE(RequestWords(caller, service, 22, {}).first == Kernel::ResultInvalidState);
    REQUIRE(RequestWords(caller, service, 11, missing_event).first == Kernel::ResultInvalidSize);
}

TEST_CASE("Native display event queues retain, coalesce, and consume notifications",
          "[core][ipc][gsv]") {
    Service::GSV::DisplayEventQueue queue;
    REQUIRE_FALSE(queue.Notify(1));
    REQUIRE(queue.Initialize(0) == Result{ErrorModule::VI, 4});
    REQUIRE(queue.Initialize(1024) == Result{ErrorModule::VI, 4});
    REQUIRE(queue.Initialize(30) == Result{ErrorModule::VI, 4});
    REQUIRE(queue.Initialize(1023).IsSuccess());
    REQUIRE(queue.Initialize(31) == Result{ErrorModule::VI, 5});
    REQUIRE_FALSE(queue.Notify(0));
    REQUIRE_FALSE(queue.Notify(1024));
    REQUIRE_FALSE(queue.Notify(0xffffffff));
    for (const u32 id : {1U, 31U, 32U, 1023U, 31U}) {
        REQUIRE(queue.Notify(id));
    }
    std::array<u8, 128> output{};
    queue.Drain(output);
    REQUIRE(output[0] == 2);
    REQUIRE(output[3] == 0x80);
    REQUIRE(output[4] == 1);
    REQUIRE(output[127] == 0x80);
    queue.Drain(output);
    REQUIRE(std::ranges::all_of(output, [](u8 byte) { return byte == 0; }));
}

TEST_CASE("Native display notification reads mark unavailable slots and consume short reads",
          "[core][ipc][gsv]") {
    Service::GSV::DisplayEventQueue queue;
    REQUIRE(queue.Initialize(31).IsSuccess());
    REQUIRE(queue.Notify(31));
    REQUIRE_FALSE(queue.Notify(32));
    std::array<u8, 8> output{};
    queue.Drain(output);
    REQUIRE(output[3] == 0x80);
    REQUIRE(output[4] == 0xff);
    REQUIRE(output[7] == 0xff);
    REQUIRE(queue.Notify(31));
    queue.Drain(std::span{output}.first(1));
    queue.Drain(output);
    REQUIRE(output[3] == 0);
    REQUIRE(output[4] == 0xff);
}

TEST_CASE("Native display control replies expose bounded pools and reject invalid requests",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::DisplaySession service{caller.system, &caller.process};
    const auto [event_result, max_event] = RequestWords(caller, service, 1, {}, 2);
    REQUIRE(event_result.IsSuccess());
    REQUIRE(max_event == 1023);
    const auto [semaphore_result, max_semaphore] = RequestWords(caller, service, 3, {}, 2);
    REQUIRE(semaphore_result.IsSuccess());
    REQUIRE(max_semaphore == 7);
    REQUIRE(RequestWords(caller, service, 11, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 1> invalid_pool{1024};
    REQUIRE(RequestWords(caller, service, 11, invalid_pool).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 1> unaligned_pool{32};
    REQUIRE(RequestWords(caller, service, 11, unaligned_pool).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 1> invalid_semaphore{8};
    REQUIRE(RequestWords(caller, service, 31, invalid_semaphore).first ==
            Result{ErrorModule::VI, 4});
    REQUIRE(RequestWords(caller, service, 32, {}).first == Kernel::ResultInvalidSize);
    REQUIRE(RequestWords(caller, service, 32, invalid_semaphore).first ==
            Result{ErrorModule::VI, 4});
    const std::array<u32, 1> missing_semaphore{0};
    REQUIRE(RequestWords(caller, service, 32, missing_semaphore).first ==
            Result{ErrorModule::VI, 7});
    REQUIRE(RequestWords(caller, service, 21, {}).first == Result{ErrorModule::VI, 5});
    const std::array<u32, 1> notify{1};
    REQUIRE(RequestWords(caller, service, 13, notify).first.IsSuccess());
}

TEST_CASE("Native display registration validates names, notification slots, and packet sizes",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::DisplaySession service{caller.system, &caller.process};
    REQUIRE(RequestWords(caller, service, 101, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 3> empty_name{0, 0, 1};
    REQUIRE(RequestWords(caller, service, 101, empty_name).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 3> zero_slot{0x61666564, 0x00746c75, 0};
    REQUIRE(RequestWords(caller, service, 101, zero_slot).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 3> invalid_slot{0x61666564, 0x00746c75, 1024};
    REQUIRE(RequestWords(caller, service, 101, invalid_slot).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 3> unknown_name{0x74736574, 0, 1};
    REQUIRE(RequestWords(caller, service, 101, unknown_name).first == Result{ErrorModule::VI, 7});
    REQUIRE(RequestWords(caller, service, 102, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 2> default_name{0x61666564, 0x00746c75};
    REQUIRE(RequestWords(caller, service, 102, default_name).first.IsSuccess());
    REQUIRE(RequestWords(caller, service, 102, default_name).first.IsSuccess());
    const std::array<u32, 2> absent_name{};
    REQUIRE(RequestWords(caller, service, 102, absent_name).first.IsSuccess());
    REQUIRE(RequestWords(caller, service, 261, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 3> invalid_semaphore{0x61666564, 0x00746c75, 8};
    REQUIRE(RequestWords(caller, service, 261, invalid_semaphore).first ==
            Result{ErrorModule::VI, 4});
    const std::array<u32, 3> unregistered_display{0x61666564, 0x00746c75, 0};
    REQUIRE(RequestWords(caller, service, 261, unregistered_display).first ==
            Result{ErrorModule::VI, 7});
    REQUIRE(RequestWords(caller, service, 262, {}).first == Kernel::ResultInvalidSize);
    REQUIRE(RequestWords(caller, service, 262, unregistered_display).first.IsSuccess());
    REQUIRE(RequestWords(caller, service, 262, invalid_semaphore).first.IsSuccess());
}

TEST_CASE("Copied guest objects remain referenced until the IPC reply is released", "[core][ipc]") {
    Caller caller;
    const auto references = caller.process.GetReferenceCount();
    {
        Service::HLERequestContext ctx{caller.system.Kernel(), caller.process.GetMemory(), nullptr,
                                       &caller.thread};
        Kernel::KScopedAutoObject<Kernel::KAutoObject> object{caller.system.Kernel(),
                                                              &caller.process};
        REQUIRE(caller.process.GetReferenceCount() == references + 1);
        ctx.AddCopyObject(std::move(object));
        REQUIRE(object.IsNull());
        REQUIRE(caller.process.GetReferenceCount() == references + 1);
    }
    REQUIRE(caller.process.GetReferenceCount() == references);
}

TEST_CASE("Native display layers share a bounded pool with owner-checked release",
          "[core][ipc][gsv]") {
    Service::VI::NativeLayerPool pool;
    std::array<u64, Service::VI::NativeLayerPool::Capacity> ids{};
    for (size_t i = 0; i < ids.size(); ++i) {
        REQUIRE(pool.Create(&ids[i], i % 2).IsSuccess());
        REQUIRE(ids[i] != 0);
    }
    u64 extra_id = 0xfeed;
    REQUIRE(pool.Create(&extra_id, 2) == Result{ErrorModule::VI, 1});
    REQUIRE(extra_id == 0xfeed);
    REQUIRE(pool.Destroy(ids[0], 1) == Result{ErrorModule::VI, 7});
    REQUIRE(pool.Create(&extra_id, 2) == Result{ErrorModule::VI, 1});
    REQUIRE(pool.Destroy(ids[0], 0).IsSuccess());
    REQUIRE(pool.Create(&extra_id, 2).IsSuccess());
    REQUIRE(extra_id != ids[0]);
    REQUIRE(pool.Destroy(ids[0], 0) == Result{ErrorModule::VI, 7});
    REQUIRE(pool.Destroy(extra_id, 2).IsSuccess());
    REQUIRE(pool.Destroy(extra_id, 2) == Result{ErrorModule::VI, 7});
    pool.Clear();
    REQUIRE(pool.Destroy(ids[1], 1) == Result{ErrorModule::VI, 7});
    for (auto& id : ids) {
        REQUIRE(pool.Create(&id, 3).IsSuccess());
        REQUIRE(id > extra_id);
    }
}

TEST_CASE("Native display layer creation validates tokens and both notification slots",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::DisplaySession service{caller.system, &caller.process};
    REQUIRE(RequestWords(caller, service, 201, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 4> zero_token{0, 0, 1, 2};
    REQUIRE(RequestWords(caller, service, 201, zero_token).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 4> zero_state{1, 0, 0, 2};
    REQUIRE(RequestWords(caller, service, 201, zero_state).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 4> invalid_buffer{1, 0, 1, 1024};
    REQUIRE(RequestWords(caller, service, 201, invalid_buffer).first == Result{ErrorModule::VI, 4});
    REQUIRE(RequestWords(caller, service, 202, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 2> absent_token{1, 0};
    REQUIRE(RequestWords(caller, service, 202, absent_token).first.IsSuccess());
    REQUIRE(RequestWords(caller, service, 202, absent_token).first.IsSuccess());
    REQUIRE(RequestWords(caller, service, 212, {}, 3).first == Kernel::ResultInvalidSize);
    const auto [state_result, state] = RequestWords(caller, service, 212, absent_token, 3);
    REQUIRE(state_result == Result{ErrorModule::VI, 7});
    REQUIRE(state == 0);
    REQUIRE(RequestWords(caller, service, 251, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 4> missing_layer{1, 0, 0x61666564, 0x00746c75};
    REQUIRE(RequestWords(caller, service, 251, missing_layer).first == Result{ErrorModule::VI, 7});
    REQUIRE(RequestWords(caller, service, 252, {}).first == Kernel::ResultInvalidSize);
    REQUIRE(RequestWords(caller, service, 252, absent_token).first.IsSuccess());
}

TEST_CASE("Empty native layers retain their buffer state across display binding",
          "[core][ipc][gsv]") {
    Service::VI::NativeLayerPool pool;
    u64 id{};
    REQUIRE(pool.Create(&id, 4).IsSuccess());
    u64 state = 0xfeed;
    REQUIRE(pool.QueryBufferState(&state, id, 5) == Result{ErrorModule::VI, 7});
    REQUIRE(state == 0xfeed);
    REQUIRE(pool.QueryBufferState(&state, id, 4).IsSuccess());
    REQUIRE(state == 0);
    REQUIRE(pool.Bind(id, 5, 0) == Result{ErrorModule::VI, 7});
    REQUIRE(pool.Bind(id, 4, 0).IsSuccess());
    REQUIRE(pool.Bind(id, 4, 0) == Result{ErrorModule::VI, 1});
    REQUIRE(pool.Unbind(id, 5) == Result{ErrorModule::VI, 7});
    REQUIRE(pool.Bind(id, 4, 1) == Result{ErrorModule::VI, 1});
    REQUIRE(pool.Unbind(id, 4).IsSuccess());
    REQUIRE(pool.Unbind(id, 4).IsSuccess());
    REQUIRE(pool.Bind(id, 4, 1).IsSuccess());
    REQUIRE(pool.QueryBufferState(&state, id, 4).IsSuccess());
    REQUIRE(state == 0);
    REQUIRE(pool.Destroy(id, 4).IsSuccess());
    REQUIRE(pool.Bind(id, 4, 1) == Result{ErrorModule::VI, 7});
    REQUIRE(pool.Unbind(id, 4) == Result{ErrorModule::VI, 7});
}

TEST_CASE("Native layer selection replaces only a validated bound layer", "[core][ipc][gsv]") {
    Service::VI::NativeLayerPool pool;
    u64 first{}, second{};
    REQUIRE(pool.Create(&first, 4).IsSuccess());
    REQUIRE(pool.Create(&second, 4).IsSuccess());
    REQUIRE(pool.Bind(first, 4, 0).IsSuccess());
    REQUIRE(pool.Select(first, 4, 0).IsSuccess());
    bool selected{};
    REQUIRE(pool.QuerySelected(&selected, first, 4).IsSuccess());
    REQUIRE(selected);
    REQUIRE(pool.Select(second, 4, 0, first) == Result{ErrorModule::VI, 1});
    REQUIRE(pool.QuerySelected(&selected, first, 4).IsSuccess());
    REQUIRE(selected);
    REQUIRE(pool.Bind(second, 4, 0).IsSuccess());
    REQUIRE(pool.Select(second, 5, 0, first) == Result{ErrorModule::VI, 7});
    REQUIRE(pool.QuerySelected(&selected, first, 4).IsSuccess());
    REQUIRE(selected);
    REQUIRE(pool.Select(second, 4, 0, first).IsSuccess());
    REQUIRE(pool.QuerySelected(&selected, first, 4).IsSuccess());
    REQUIRE_FALSE(selected);
    REQUIRE(pool.QuerySelected(&selected, second, 4).IsSuccess());
    REQUIRE(selected);
    REQUIRE(pool.Select(second, 4, 0, second).IsSuccess());
    REQUIRE(pool.Unbind(second, 4).IsSuccess());
    REQUIRE(pool.QuerySelected(&selected, second, 4).IsSuccess());
    REQUIRE_FALSE(selected);
    REQUIRE(pool.Destroy(second, 4).IsSuccess());
    REQUIRE(pool.QuerySelected(&selected, second, 4) == Result{ErrorModule::VI, 7});
}

TEST_CASE("Native display selection validates both names and the layer token", "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::DisplaySession service{caller.system, &caller.process};
    REQUIRE(RequestWords(caller, service, 111, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 6> no_token{0x61666564, 0x00746c75, 0x61666564, 0x00746c75, 0, 0};
    REQUIRE(RequestWords(caller, service, 111, no_token).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 6> no_slot{0x61666564, 0x00746c75, 0, 0, 1, 0};
    REQUIRE(RequestWords(caller, service, 111, no_slot).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 6> missing_layer{0x61666564, 0x00746c75, 0x61666564, 0x00746c75, 1, 0};
    REQUIRE(RequestWords(caller, service, 111, missing_layer).first == Result{ErrorModule::VI, 7});
}

TEST_CASE("Native timestamp recorder setup validates bounds and preserves initialization",
          "[core][ipc][gsv]") {
    Caller caller;
    Service::GSV::DisplaySession service{caller.system, &caller.process};
    const auto [result, max_id] = RequestWords(caller, service, 5, {}, 2);
    REQUIRE(result.IsSuccess());
    REQUIRE(max_id == 7);
    REQUIRE(RequestWords(caller, service, 51, {}).first == Kernel::ResultInvalidSize);
    const std::array<u32, 1> invalid_count{8};
    REQUIRE(RequestWords(caller, service, 51, invalid_count).first == Result{ErrorModule::VI, 4});
    const std::array<u32, 1> full_count{7};
    REQUIRE(RequestWords(caller, service, 51, full_count).first.IsSuccess());
    REQUIRE(RequestWords(caller, service, 51, full_count).first == Result{ErrorModule::VI, 5});
}
