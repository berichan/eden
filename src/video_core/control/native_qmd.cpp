// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <fmt/format.h>
#include "common/logging.h"
#include "video_core/control/native_qmd.h"

namespace Tegra::Control {
namespace {

bool AllowQmdDiagnostic() {
    static std::atomic<u32> emitted{};
    u32 count = emitted.load(std::memory_order_relaxed);
    while (count < 16384) {
        if (emitted.compare_exchange_weak(count, count + 1, std::memory_order_relaxed))
            return true;
    }
    return false;
}

bool ValidDescriptor(const NativeQmdScheduler::Descriptor& descriptor) {
    return (descriptor[18] & 0xff) == 0x30 && !(descriptor[4] & 0x40000100) &&
           !(descriptor[16] & (1U << 5));
}

bool ValidRelease(u32 flags) {
    const u32 size = flags >> 30;
    const bool wide_payload = (flags & (1U << 29)) != 0;
    // Reserved fields, traps and the reserved structure-size encoding have no fallback.
    return !(flags & (0x7ffU << 8)) && !(flags & (3U << 27)) && size != 3 &&
           !(size == 1 && wide_payload) && ((flags >> 24) & 3) <= 1;
}

} // namespace

NativeQmdScheduler::Result NativeQmdScheduler::Signal(GPUVAddr address, u32 argument,
                                                      const Host& host) {
    if (HasFailed()) {
        return failure;
    }
    Result result;
    const auto fail = [&](Error error, GPUVAddr failed_address) {
        if (AllowQmdDiagnostic())
            LOG_DEBUG(HW_GPU, "Native QMD failure: scheduler={}, node={:#x}, error={}",
                      fmt::ptr(this), failed_address, static_cast<u32>(error));
        result.error = error;
        result.address = failed_address;
        failure = result;
        return result;
    };
    if ((address & 0xff) || address >= (u64{1} << 40) || (argument & ~0x00ffc00fU)) {
        return fail(Error::InvalidRequest, address);
    }
    struct Request {
        GPUVAddr address;
        u32 action;
        u32 count;
    };
    std::array<Request, MaxChain> requests;
    requests[0] = {address, argument & 15, ((argument >> 14) & 0x3ff) + 1};
    size_t request_count = 1;
    for (size_t index = 0; index < request_count; ++index) {
        const auto request = requests[index];
        if (request.count != 1 && request.action != 7 && request.action != 12) {
            return fail(Error::InvalidRequest, request.address);
        }
        if (host.should_stop && host.should_stop(host.context)) {
            return fail(Error::Cancelled, request.address);
        }
        if (request.action == 0) {
            continue;
        }
        if (request.action != 1 && request.action != 2 && request.action != 3 &&
            request.action != 7 && request.action != 12) {
            return fail(Error::UnsupportedAction, request.address);
        }
        if (states.size() >= MaxDescriptors && !states.contains(request.address)) {
            return fail(Error::DescriptorLimit, request.address);
        }
        auto& state = states[request.address];
        if (AllowQmdDiagnostic())
            LOG_DEBUG(HW_GPU,
                      "Native QMD request: scheduler={}, node={:#x}, action={}, count={}, "
                      "loaded={}, scheduled={}, issued={}, dependencies={}, deferred={}",
                      fmt::ptr(this), request.address, request.action, request.count, state.loaded,
                      state.scheduled, state.issued, state.dependencies, state.deferred);
        if (request.action == 1) {
            if (state.scheduled && !state.issued) {
                return fail(Error::UnsupportedAction, request.address);
            }
            state.loaded = false;
            state.scheduled = false;
            state.issued = false;
            if (AllowQmdDiagnostic())
                LOG_DEBUG(HW_GPU,
                          "Native QMD invalidated: scheduler={}, node={:#x}, dependencies={}, "
                          "deferred={}",
                          fmt::ptr(this), request.address, state.dependencies, state.deferred);
            continue;
        }
        if (request.action == 2 || request.action == 3) {
            if (request.action == 3 && state.scheduled && !state.issued) {
                return fail(Error::UnsupportedAction, request.address);
            }
            if (request.action == 2 && state.issued) {
                return fail(Error::UnsupportedAction, request.address);
            }
            if (request.action == 3 || !state.loaded) {
                Descriptor descriptor;
                if (!host.read || !host.read(host.context, request.address, descriptor)) {
                    return fail(Error::ReadFault, request.address);
                }
                if (!ValidDescriptor(descriptor)) {
                    return fail(Error::UnsupportedDescriptor, request.address);
                }
                const s64 dependencies = ((descriptor[4] >> 14) & 0xffff) + state.deferred;
                if (dependencies > CounterLimit) {
                    return fail(Error::CounterOverflow, request.address);
                }
                state.descriptor = descriptor;
                state.dependencies = dependencies;
                state.deferred = 0;
                state.loaded = true;
                state.issued = false;
                if (AllowQmdDiagnostic())
                    LOG_DEBUG(
                        HW_GPU,
                        "Native QMD loaded: scheduler={}, node={:#x}, word4={:#x}, initial={}, "
                        "dependencies={}, grid={}x{}x{}, program={:#x}, dependent={:#x}, "
                        "dependent_enable={}, dependent_action={}",
                        fmt::ptr(this), request.address, descriptor[4],
                        (descriptor[4] >> 14) & 0xffff, state.dependencies, descriptor[12],
                        descriptor[13] & 0xffff, descriptor[14] & 0xffff,
                        (GPUVAddr{descriptor[49] & 0x1ffff} << 32) | descriptor[48],
                        GPUVAddr{descriptor[15]} << 8, descriptor[16] & 1,
                        (descriptor[16] >> 1) & 7);
            }
            state.scheduled = true;
        } else {
            const s64 delta = request.action == 7 ? -s64{request.count} : s64{request.count};
            auto& counter = !state.loaded || state.issued ? state.deferred : state.dependencies;
            if (counter + delta > CounterLimit || counter + delta < -CounterLimit) {
                return fail(Error::CounterOverflow, request.address);
            }
            counter += delta;
        }
        if (!state.loaded || !state.scheduled || state.issued || state.dependencies > 0) {
            if (AllowQmdDiagnostic())
                LOG_DEBUG(HW_GPU,
                          "Native QMD parked: scheduler={}, node={:#x}, loaded={}, scheduled={}, "
                          "issued={}, dependencies={}, deferred={}",
                          fmt::ptr(this), request.address, state.loaded, state.scheduled,
                          state.issued, state.dependencies, state.deferred);
            continue;
        }
        if (state.dependencies < 0) {
            return fail(Error::CounterUnderflow, request.address);
        }
        Descriptor current;
        if (!host.read || !host.read(host.context, request.address, current)) {
            return fail(Error::ReadFault, request.address);
        }
        if (!ValidDescriptor(current)) {
            return fail(Error::UnsupportedDescriptor, request.address);
        }
        // The exact native cache/fetch partition is not established. Support patched
        // releases, but reject changed launch/scheduling fields before publishing work.
        for (size_t word = 0; word < current.size(); ++word) {
            if ((word >= 24 && word < 32) || (word >= 52 && word < 56)) {
                continue;
            }
            if (current[word] != state.descriptor[word]) {
                LOG_ERROR(HW_GPU, "Native QMD changed unsupported field: address={:#x}, word={}",
                          request.address, word);
                return fail(Error::UnsupportedDescriptor, request.address);
            }
        }
        std::array<Release, 3> releases;
        std::array<u32, 3> release_words;
        size_t release_count{};
        for (u32 word : {24U, 28U, 52U}) {
            const u32 flags = current[word + 1];
            if (!(flags & (1U << 23))) {
                continue;
            }
            if (!ValidRelease(flags)) {
                return fail(Error::UnsupportedRelease, request.address);
            }
            const Release release{(GPUVAddr{flags & 0xff} << 32) | current[word],
                                  (u64{current[word + 3]} << 32) | current[word + 2], flags};
            if (!host.validate_release || !host.validate_release(host.context, release)) {
                return fail(Error::InvalidRelease, request.address);
            }
            release_words[release_count] = word;
            releases[release_count++] = release;
        }
        const auto& descriptor = state.descriptor;
        u32 dependent_action{};
        if (descriptor[16] & 1) {
            switch ((descriptor[16] >> 1) & 7) {
            case 1:
                dependent_action = 2;
                break;
            case 3:
                dependent_action = 3;
                break;
            case 4:
                dependent_action = 7;
                break;
            default:
                return fail(Error::UnsupportedAction, request.address);
            }
            if (request_count == requests.size()) {
                return fail(Error::ChainLimit, request.address);
            }
        }
        if (AllowQmdDiagnostic())
            LOG_DEBUG(HW_GPU,
                      "Native QMD ready: scheduler={}, node={:#x}, grid={}x{}x{}, program={:#x}, "
                      "dependent={:#x}, action={}, releases={}",
                      fmt::ptr(this), request.address, descriptor[12], descriptor[13] & 0xffff,
                      descriptor[14] & 0xffff,
                      (GPUVAddr{descriptor[49] & 0x1ffff} << 32) | descriptor[48],
                      GPUVAddr{descriptor[15]} << 8, dependent_action, release_count);
        if (descriptor[12] && (descriptor[13] & 0xffff) && (descriptor[14] & 0xffff)) {
            if (host.should_stop && host.should_stop(host.context)) {
                return fail(Error::Cancelled, request.address);
            }
            if (!host.dispatch || !host.dispatch(host.context, descriptor)) {
                return fail(Error::DispatchFailed, request.address);
            }
            ++result.dispatches;
        }
        for (const auto& release : std::span{releases}.first(release_count)) {
            if (host.should_stop && host.should_stop(host.context)) {
                return fail(Error::Cancelled, request.address);
            }
            if (!host.release || !host.release(host.context, release)) {
                return fail(Error::ReleaseFailed, request.address);
            }
            if (AllowQmdDiagnostic()) {
                const u32 word = release_words[static_cast<size_t>(&release - releases.data())];
                const auto& cached = state.descriptor;
                const bool patched =
                    cached[word] != current[word] || cached[word + 1] != current[word + 1] ||
                    cached[word + 2] != current[word + 2] || cached[word + 3] != current[word + 3];
                if (patched)
                    LOG_DEBUG(HW_GPU,
                              "Native QMD release issued: scheduler={}, node={:#x}, slot_word={}, "
                              "address={:#x}, payload={:#x}, flags={:#x}, cached_address={:#x}, "
                              "cached_payload={:#x}, cached_flags={:#x}",
                              fmt::ptr(this), request.address, word, release.address,
                              release.payload, release.flags,
                              (GPUVAddr{cached[word + 1] & 0xff} << 32) | cached[word],
                              (u64{cached[word + 3]} << 32) | cached[word + 2], cached[word + 1]);
                else
                    LOG_DEBUG(HW_GPU,
                              "Native QMD release issued: scheduler={}, node={:#x}, slot_word={}, "
                              "address={:#x}, payload={:#x}, flags={:#x}",
                              fmt::ptr(this), request.address, word, release.address,
                              release.payload, release.flags);
            }
            ++result.releases;
        }
        state.issued = true;
        if (AllowQmdDiagnostic())
            LOG_DEBUG(HW_GPU,
                      "Native QMD issued: scheduler={}, node={:#x}, dependencies={}, deferred={}, "
                      "signal_dispatches={}, signal_releases={}",
                      fmt::ptr(this), request.address, state.dependencies, state.deferred,
                      result.dispatches, result.releases);
        if (dependent_action) {
            requests[request_count++] = {GPUVAddr{descriptor[15]} << 8, dependent_action, 1};
        }
    }
    return result;
}

bool NativeQmdScheduler::HasPendingDispatch() const {
    return std::ranges::any_of(
        states, [](const auto& entry) { return entry.second.scheduled && !entry.second.issued; });
}

bool NativeQmdScheduler::HasFailed() const {
    return failure.error != Error::None;
}

} // namespace Tegra::Control
