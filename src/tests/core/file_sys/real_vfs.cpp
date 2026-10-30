// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef _WIN32
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/fs/file.h"
#include "common/fs/path_util.h"
#include "common/scope_exit.h"
#include "core/file_sys/vfs/vfs_real.h"

TEST_CASE("Real VFS evicts and reopens files below the Windows stream limit", "[filesystem]") {
    namespace fs = std::filesystem;
    const auto directory =
        fs::temp_directory_path() /
        ("eden-vfs-streams-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(fs::create_directory(directory));
    SCOPE_EXIT {
        std::error_code error;
        fs::remove_all(directory, error);
    };
    const int original_limit = _getmaxstdio();
    REQUIRE(_setmaxstdio(128) == 128);
    SCOPE_EXIT {
        _setmaxstdio(original_limit);
    };
    FileSys::RealVfsFilesystem filesystem;
    std::vector<FileSys::VirtualFile> files;
    for (u32 index = 0; index < 160; ++index) {
        const auto path = directory / std::to_string(index);
        {
            Common::FS::IOFile output{path, Common::FS::FileAccessMode::Write};
            REQUIRE(output.IsOpen());
            REQUIRE(output.WriteSpan<u32>(std::span{&index, 1}) == 1);
        }
        auto file = filesystem.OpenFile(Common::FS::PathToUTF8String(path));
        REQUIRE(file != nullptr);
        u32 actual{};
        REQUIRE(file->Read(reinterpret_cast<u8*>(&actual), sizeof(actual), 0) == sizeof(actual));
        REQUIRE(actual == index);
        files.push_back(std::move(file));
    }
    for (u32 index = 0; index < files.size(); ++index) {
        u32 actual{};
        REQUIRE(files[index]->Read(reinterpret_cast<u8*>(&actual), sizeof(actual), 0) ==
                sizeof(actual));
        REQUIRE(actual == index);
    }
}
#endif
