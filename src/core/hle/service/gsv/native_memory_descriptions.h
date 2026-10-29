// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/result.h"
#include "core/hle/service/gsv/gpu_memory_pool.h"

namespace Service::GSV {

class NativeMemoryDescriptions {
public:
    static constexpr Result InvalidArgument{0x100f3};
    static constexpr Result InvalidHandle{0x280f3};
    static constexpr Result OutOfMemory{0x80f3};
    static constexpr Result Busy{0x180f3};

    struct Binding {
        u64 address;
        u64 offset;
        u32 memory_handle;
    };

    struct Description {
        u64 size;
        u8 type;
        std::optional<Binding> binding;
    };

    ~NativeMemoryDescriptions() {
        for (auto& [handle, record] : m_descriptions) {
            if (record.pool) {
                const auto result =
                    record.pool->ReleaseReference(record.description.binding->memory_handle);
                if (result.IsError()) {
                    LOG_ERROR(
                        Service,
                        "Native memory description teardown failed: handle={:#x}, result={:#x}",
                        handle, result.raw);
                }
            }
        }
    }

    Result Create(u32* out_handle, u8 type, u64 size) {
        R_UNLESS(type <= 3 && size != 0, InvalidArgument);
        const std::scoped_lock lock{m_mutex};
        R_UNLESS(m_descriptions.size() < std::numeric_limits<u32>::max(), OutOfMemory);
        u32 handle = m_last_handle;
        do {
            ++handle;
            if (handle == 0)
                ++handle;
        } while (m_descriptions.contains(handle));
        try {
            m_descriptions.emplace(handle, Record{Description{size, type, std::nullopt}, {}});
        } catch (const std::bad_alloc&) {
            R_RETURN(OutOfMemory);
        }
        m_last_handle = handle;
        *out_handle = handle;
        R_SUCCEED();
    }

    Result Destroy(u32 handle) {
        const std::scoped_lock lock{m_mutex};
        const auto it = m_descriptions.find(handle);
        R_UNLESS(it != m_descriptions.end(), InvalidHandle);
        R_UNLESS(!it->second.image_attached, Busy);
        if (it->second.pool) {
            R_TRY(it->second.pool->ReleaseReference(it->second.description.binding->memory_handle));
        }
        m_descriptions.erase(it);
        R_SUCCEED();
    }

    Result Bind(u32 handle, std::shared_ptr<GpuMemoryPool> pool, u64 address, u64 size,
                u32 memory_handle, u64 offset) {
        const std::scoped_lock lock{m_mutex};
        const auto it = m_descriptions.find(handle);
        R_UNLESS(it != m_descriptions.end(), InvalidHandle);
        auto& record = it->second;
        R_UNLESS(!record.description.binding, InvalidHandle);
        R_UNLESS(record.description.type == 3 && size != 0 && size <= record.description.size,
                 InvalidArgument);
        // Partial native blocks need a range list and completion accounting before use.
        R_UNLESS(size == record.description.size, Kernel::ResultNotImplemented);
        R_UNLESS(pool, Kernel::ResultInvalidState);
        R_TRY(pool->RetainReference(memory_handle, address, size, offset));
        record.description.binding = Binding{address, offset, memory_handle};
        record.pool = std::move(pool);
        R_SUCCEED();
    }

    std::optional<Description> Get(u32 handle) const {
        const std::scoped_lock lock{m_mutex};
        const auto it = m_descriptions.find(handle);
        return it != m_descriptions.end() ? std::optional{it->second.description} : std::nullopt;
    }

    Result AcquireImage(Description* out, u32 handle, u64 size) {
        R_UNLESS(size != 0, InvalidArgument);
        const std::scoped_lock lock{m_mutex};
        const auto it = m_descriptions.find(handle);
        R_UNLESS(it != m_descriptions.end(), InvalidHandle);
        auto& record = it->second;
        R_UNLESS(record.description.binding && !record.image_attached &&
                     size <= record.description.size,
                 Busy);
        record.image_attached = true;
        *out = record.description;
        R_SUCCEED();
    }

    void ReleaseImage(u32 handle) {
        const std::scoped_lock lock{m_mutex};
        const auto it = m_descriptions.find(handle);
        ASSERT(it != m_descriptions.end() && it->second.image_attached);
        it->second.image_attached = false;
    }

private:
    struct Record {
        Description description;
        std::shared_ptr<GpuMemoryPool> pool;
        bool image_attached{};
    };

    // Unbound descriptions own no address. Bound records retain the registered allocation's
    // existing physical pin, independently of its GPU mappings, until destruction.
    mutable std::mutex m_mutex;
    std::map<u32, Record> m_descriptions;
    u32 m_last_handle{};
};

} // namespace Service::GSV
