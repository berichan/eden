// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <new>
#include "core/hle/service/gsv/native_image_pool.h"

namespace Service::GSV {

NativeImage::NativeImage(const NativeImagePreference& preference_, const NativeImageLayout& layout_)
    : preference{preference_}, layout{layout_} {}

NativeImage::~NativeImage() {
    if (m_owner) {
        m_owner->ReleaseImage(m_memory_handle);
    }
}

Result NativeImage::ResolveBacking(DAddr* out_address) const {
    R_UNLESS(m_owner, Kernel::ResultInvalidState);
    R_RETURN(m_owner->ResolveImageBacking(out_address, m_memory_handle, layout.size));
}

NativeImagePool::NativeImagePool(std::shared_ptr<NativeMemoryDescriptions> descriptions)
    : m_descriptions{std::move(descriptions)} {}

Result NativeImagePool::Create(u32* out_handle, const NativeImagePreference& preference, u32 layers,
                               u32 memory_handle) {
    NativeImageLayout layout;
    R_TRY(CalculateNativeImageLayout(&layout, preference, layers));
    // Array images require a range of public handles and per-layer import descriptors.
    R_UNLESS(layers == 1, Kernel::ResultNotImplemented);
    R_UNLESS(u64{layout.width} * layout.bytes_per_pixel <= layout.row_pitch,
             NativeMemoryDescriptions::InvalidArgument);
    R_UNLESS(m_descriptions, Kernel::ResultInvalidState);
    const std::scoped_lock lock{m_mutex};
    R_UNLESS(m_images.size() < std::numeric_limits<u32>::max(),
             NativeMemoryDescriptions::OutOfMemory);
    u32 handle = m_last_handle;
    do {
        ++handle;
        if (handle == 0)
            ++handle;
    } while (m_images.contains(handle));

    try {
        auto image = std::make_shared<NativeImage>(preference, layout);
        auto it = m_images.emplace(handle, image).first;
        const auto result =
            m_descriptions->AcquireImage(&image->m_memory, memory_handle, layout.size);
        if (result.IsError()) {
            m_images.erase(it);
            R_RETURN(result);
        }
        // All host allocations precede the lease, so a failed creation never strands a busy block.
        image->m_memory_handle = memory_handle;
        image->m_owner = m_descriptions;
        m_last_handle = handle;
        *out_handle = handle;
        R_SUCCEED();
    } catch (const std::bad_alloc&) {
        R_RETURN(NativeMemoryDescriptions::OutOfMemory);
    }
}

Result NativeImagePool::Destroy(u32 handle) {
    const std::scoped_lock lock{m_mutex};
    const auto it = m_images.find(handle);
    R_UNLESS(it != m_images.end(), NativeMemoryDescriptions::InvalidHandle);
    R_UNLESS(it->second.use_count() == 1, NativeMemoryDescriptions::Busy);
    m_images.erase(it);
    R_SUCCEED();
}

std::shared_ptr<const NativeImage> NativeImagePool::Acquire(u32 handle) const {
    const std::scoped_lock lock{m_mutex};
    const auto it = m_images.find(handle);
    return it == m_images.end() ? nullptr : it->second;
}

Result NativeImagePool::RegisterAlias(u32 handle, u32 count) {
    const std::scoped_lock lock{m_mutex};
    const auto image = m_images.find(handle);
    R_UNLESS(image != m_images.end(), NativeMemoryDescriptions::InvalidHandle);
    // A single-layer image contributes exactly one handle to a native alias pool.
    R_UNLESS(count == 1, NativeMemoryDescriptions::InvalidArgument);
    const auto alias = m_aliases.find(handle);
    if (alias != m_aliases.end()) {
        R_UNLESS(alias->second.registrations < 255, NativeMemoryDescriptions::OutOfMemory);
        ++alias->second.registrations;
        R_SUCCEED();
    }
    try {
        m_aliases.emplace(handle, Alias{image->second, 1});
    } catch (const std::bad_alloc&) {
        R_RETURN(NativeMemoryDescriptions::OutOfMemory);
    }
    R_SUCCEED();
}

Result NativeImagePool::UnregisterAlias(u32 handle) {
    const std::scoped_lock lock{m_mutex};
    const auto alias = m_aliases.find(handle);
    R_UNLESS(alias != m_aliases.end(), NativeMemoryDescriptions::InvalidHandle);
    if (--alias->second.registrations == 0) {
        m_aliases.erase(alias);
    }
    R_SUCCEED();
}

std::shared_ptr<const NativeImage> NativeImagePool::AcquireAlias(u32 handle) const {
    const std::scoped_lock lock{m_mutex};
    const auto alias = m_aliases.find(handle);
    return alias == m_aliases.end() ? nullptr : alias->second.image;
}

void NativeImagePool::ClearAliases() {
    const std::scoped_lock lock{m_mutex};
    m_aliases.clear();
}

} // namespace Service::GSV
