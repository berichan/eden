// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include "video_core/control/native_gpu_context.h"
#include "video_core/control/native_inline_upload.h"
#include "video_core/control/native_macro_processor.h"

namespace Tegra::Control {

// Word methods from NVIDIA classes/compute/clc7c0.h. Native registers must not alias
// the older compute class's action registers.
class NativeCompute {
public:
    struct Host {
        NativeGpuGraphicsContext::QmdBackend qmd;
        bool (*barrier)(void*){};
    };
    enum class Error {
        None,
        InvalidMethod,
        UnsupportedMethod,
        MacroFailed,
        QmdFailed,
        Cancelled,
        UploadFailed
    };

    explicit NativeCompute(std::shared_ptr<const NativeGpuGraphicsContext> context);
    bool CallMethod(u32 method, u32 value, bool last, const Host& host);
    u32 ReadRegister(u32 method) const;
    Error GetError() const;
    bool HasPendingMacro() const;
    bool HasPendingInlineUpload() const;
    const NativeQmdScheduler::Result& GetQmdResult() const;

private:
    bool ReadMethod(u32 method, u32& value) const;
    bool WriteMethod(u32 method, u32 value, const Host& host);
    bool Fail(Error error);

    const std::shared_ptr<const NativeGpuGraphicsContext> context;
    NativeMacroProcessor macros;
    NativeInlineUpload inline_upload;
    std::array<u32, NativeMacroProcessor::MacroMethodStart> registers{};
    Error error{};
    NativeQmdScheduler::Result qmd_result;
};

} // namespace Tegra::Control
