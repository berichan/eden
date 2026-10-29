// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include "video_core/control/native_qmd.h"

namespace Tegra::Control {
namespace {

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
        if (request.action == 1) {
            if (state.scheduled && !state.issued) {
                return fail(Error::UnsupportedAction, request.address);
            }
            state.loaded = false;
            state.scheduled = false;
            state.issued = false;
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
            continue;
        }
        if (state.dependencies < 0) {
            return fail(Error::CounterUnderflow, request.address);
        }
        Descriptor current;
        if (!host.read || !host.read(host.context, request.address, current)) {
            return fail(Error::ReadFault, request.address);
        }
        std::array<Release, 3> releases;
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
        if (descriptor[12] && (descriptor[13] & 0xffff) && (descriptor[14] & 0xffff)) {
            if (!host.dispatch || !host.dispatch(host.context, descriptor)) {
                return fail(Error::DispatchFailed, request.address);
            }
            ++result.dispatches;
        }
        for (const auto& release : std::span{releases}.first(release_count)) {
            if (!host.release || !host.release(host.context, release)) {
                return fail(Error::ReleaseFailed, request.address);
            }
            ++result.releases;
        }
        state.issued = true;
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
