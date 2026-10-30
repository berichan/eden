// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include "core/hle/service/gsv/native_image_layout.h"
#include "core/hle/service/gsv/native_memory_descriptions.h"

namespace Service::GSV {

class NativeImagePool;

class NativeImage {
public:
    NativeImage(const NativeImagePreference& preference_, const NativeImageLayout& layout_);
    ~NativeImage();
    NativeImage(const NativeImage&) = delete;
    NativeImage& operator=(const NativeImage&) = delete;

    const NativeMemoryDescriptions::Description& Memory() const {
        return m_memory;
    }

    // Retain this image through cache synchronization and the checked snapshot.
    // Memory().binding->address is CPU VA; this resolves the actual device range.
    Result ResolveBacking(DAddr* out_address) const;

    const NativeImagePreference preference;
    const NativeImageLayout layout;

private:
    friend class NativeImagePool;
    std::shared_ptr<NativeMemoryDescriptions> m_owner;
    NativeMemoryDescriptions::Description m_memory{};
    u32 m_memory_handle{};
};

class NativeImagePool {
public:
    explicit NativeImagePool(std::shared_ptr<NativeMemoryDescriptions> descriptions);

    Result Create(u32* out_handle, const NativeImagePreference& preference, u32 layers,
                  u32 memory_handle);
    Result Destroy(u32 handle);
    std::shared_ptr<const NativeImage> Acquire(u32 handle) const;
    Result RegisterAlias(u32 handle, u32 count);
    Result UnregisterAlias(u32 handle);
    std::shared_ptr<const NativeImage> AcquireAlias(u32 handle) const;
    void ClearAliases();

private:
    std::shared_ptr<NativeMemoryDescriptions> m_descriptions;
    mutable std::mutex m_mutex;
    std::map<u32, std::shared_ptr<NativeImage>> m_images;
    struct Alias {
        std::shared_ptr<const NativeImage> image;
        u8 registrations;
    };
    std::map<u32, Alias> m_aliases;
    u32 m_last_handle{};
};

} // namespace Service::GSV
