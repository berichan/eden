// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_funcs.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_offset.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/loader/deconstructed_rom_directory.h"
#include "core/loader/loader.h"
#include "core/loader/nso.h"

namespace {

using FileSys::VectorVfsDirectory;
using FileSys::VectorVfsFile;
using Loader::FileType;
using Loader::ResultStatus;

std::vector<u8> MakeNso() {
    Loader::NSOHeader header{};
    header.magic = Common::MakeMagic('N', 'S', 'O', '0');
    std::vector<u8> data(0x1000);
    std::memcpy(data.data(), &header, sizeof(header));
    return data;
}

template <typename T>
void WriteField(std::vector<u8>& data, size_t offset, T value) {
    std::memcpy(data.data() + offset, &value, sizeof(value));
}

std::vector<u8> MakeMetadata(size_t prefix_size = 0x200) {
    const size_t acid_offset = 0x80;
    const size_t acid_header = acid_offset + prefix_size;
    const size_t acid_size = prefix_size + 0x70;
    const size_t aci_offset = acid_offset + acid_size;
    std::vector<u8> data(aci_offset + 0x60);
    std::memcpy(data.data(), "META", 4);
    data[0xc] = 7;
    WriteField(data, 0x70, u32_le{static_cast<u32>(aci_offset)});
    WriteField(data, 0x74, u32_le{0x60});
    WriteField(data, 0x78, u32_le{static_cast<u32>(acid_offset)});
    WriteField(data, 0x7c, u32_le{static_cast<u32>(acid_size)});
    std::memcpy(data.data() + acid_header, "ACID", 4);
    WriteField(data, acid_header + 0xc, u32_le{0x9});
    WriteField(data, acid_header + 0x20, u32_le{static_cast<u32>(prefix_size + 0x40)});
    WriteField(data, acid_header + 0x24, u32_le{0x2c});
    WriteField(data, acid_header + 0x40, u32_le{1});
    WriteField(data, acid_header + 0x44, u64_le{0x41});
    std::memcpy(data.data() + aci_offset, "ACI0", 4);
    WriteField(data, aci_offset + 0x10, u64_le{0x0100000000000000});
    WriteField(data, aci_offset + 0x20, u32_le{0x40});
    WriteField(data, aci_offset + 0x24, u32_le{0x1c});
    WriteField(data, aci_offset + 0x30, u32_le{0x5c});
    WriteField(data, aci_offset + 0x34, u32_le{4});
    WriteField(data, aci_offset + 0x40, u32_le{1});
    WriteField(data, aci_offset + 0x44, u64_le{0x21});
    WriteField(data, aci_offset + 0x5c, u32_le{0x30043f7});
    return data;
}

std::shared_ptr<VectorVfsDirectory> MakeExeFS(FileSys::VirtualDir parent = nullptr,
                                              size_t prefix_size = 0x200) {
    return std::make_shared<VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{
            std::make_shared<VectorVfsFile>(MakeNso(), "main"),
            std::make_shared<VectorVfsFile>(MakeMetadata(prefix_size), "main.npdm"),
            std::make_shared<VectorVfsFile>(MakeNso(), "sdk"),
        },
        std::vector<FileSys::VirtualDir>{}, "exefs", std::move(parent));
}

} // namespace

TEST_CASE("Program metadata supports both ACID signature sizes", "[core][loader]") {
    for (const size_t prefix_size : {0x200, 0x300}) {
        CAPTURE(prefix_size);
        FileSys::ProgramMetadata metadata;
        REQUIRE(metadata.Load(std::make_shared<VectorVfsFile>(MakeMetadata(prefix_size))) ==
                ResultStatus::Success);
        REQUIRE(metadata.GetCpuProfile() == (prefix_size == 0x300
                                                 ? Core::Hardware::CpuProfile::Ounce
                                                 : Core::Hardware::CpuProfile::Nx));
        REQUIRE(metadata.GetTitleID() == 0x0100000000000000);
        REQUIRE(metadata.GetFilesystemPermissions() == 0x21);
        REQUIRE(metadata.GetPoolPartition() == FileSys::PoolPartition::System);
        REQUIRE(metadata.GetAddressSpaceType() == FileSys::ProgramAddressSpaceType::Is39Bit);
        REQUIRE(metadata.GetKernelCapabilities() == std::vector<u32>{0x30043f7});
    }
}

TEST_CASE("Program metadata rejects invalid section ranges", "[core][loader]") {
    auto bytes = MakeMetadata(0x300);
    ResultStatus expected{};
    SECTION("Truncated ACID") {
        WriteField(bytes, 0x7c, u32_le{0x33f});
        expected = ResultStatus::ErrorBadACIDHeader;
    }
    SECTION("Missing ACID magic") {
        bytes[0x380] = 0;
        expected = ResultStatus::ErrorBadACIDHeader;
    }
    SECTION("File access control outside ACID") {
        WriteField(bytes, 0x3a0, u32_le{0xfffffff0});
        expected = ResultStatus::ErrorBadFileAccessControl;
    }
    SECTION("Kernel capabilities outside ACI") {
        WriteField(bytes, 0x420, u32_le{0xfffffff0});
        expected = ResultStatus::ErrorBadKernelCapabilityDescriptors;
    }
    SECTION("Partial kernel capability") {
        WriteField(bytes, 0x424, u32_le{3});
        expected = ResultStatus::ErrorBadKernelCapabilityDescriptors;
    }
    FileSys::ProgramMetadata metadata;
    REQUIRE(metadata.Load(std::make_shared<VectorVfsFile>(std::move(bytes))) == expected);
}

TEST_CASE("Extracted main selects the directory loader", "[core][loader]") {
    const auto exefs = MakeExeFS();
    const auto main = std::make_shared<VectorVfsFile>(MakeNso(), "main", exefs);
    REQUIRE(Loader::IdentifyFile(main) == FileType::DeconstructedRomDirectory);

    const auto sdk = std::make_shared<VectorVfsFile>(MakeNso(), "sdk", exefs);
    REQUIRE(Loader::IdentifyFile(sdk) == FileType::NSO);

    const auto standalone = std::make_shared<VectorVfsFile>(MakeNso(), "main");
    REQUIRE(Loader::IdentifyFile(standalone) == FileType::NSO);

    const auto incomplete = std::make_shared<VectorVfsDirectory>();
    const auto missing_metadata = std::make_shared<VectorVfsFile>(MakeNso(), "main", incomplete);
    incomplete->AddFile(std::make_shared<VectorVfsFile>(MakeNso(), "main"));
    REQUIRE(Loader::IdentifyFile(missing_metadata) == FileType::NSO);

    const auto invalid = std::make_shared<VectorVfsFile>(std::vector<u8>{}, "main", exefs);
    REQUIRE(Loader::AppLoader_DeconstructedRomDirectory::IdentifyType(invalid) == FileType::Error);
}

TEST_CASE("Directory loaders expose metadata before loading", "[core][loader]") {
    const auto exefs = MakeExeFS();
    const auto main = std::make_shared<VectorVfsFile>(MakeNso(), "main", exefs);
    Loader::AppLoader_DeconstructedRomDirectory from_file{main};
    Loader::AppLoader_DeconstructedRomDirectory from_directory{exefs};
    u64 program_id{};
    REQUIRE(from_file.ReadProgramId(program_id) == ResultStatus::Success);
    REQUIRE(program_id == 0x0100000000000000);
    REQUIRE(from_directory.ReadProgramId(program_id) == ResultStatus::Success);
    REQUIRE(program_id == 0x0100000000000000);
}

TEST_CASE("Directory loader discovers RomFS without copying file data", "[core][loader]") {
    const std::vector<u8> payload{1, 2, 3, 4, 5};
    const auto romfs_dir = std::make_shared<VectorVfsDirectory>(
        std::vector<FileSys::VirtualFile>{std::make_shared<VectorVfsFile>(payload, "data.bin")},
        std::vector<FileSys::VirtualDir>{}, "romfs");
    std::shared_ptr<VectorVfsDirectory> exefs;
    FileSys::VirtualFile packed;

    SECTION("RomFS image alongside executable modules") {
        exefs = MakeExeFS();
        const auto image = FileSys::CreateRomFS(romfs_dir);
        REQUIRE(image != nullptr);
        packed = std::make_shared<FileSys::OffsetVfsFile>(image, image->GetSize(), 0, "data.romfs");
        exefs->AddFile(packed);
    }

    SECTION("RomFS directory inside the executable directory") {
        exefs = MakeExeFS();
        exefs->AddDirectory(romfs_dir);
    }

    SECTION("Sibling ExeFS and RomFS directories") {
        const auto parent = std::make_shared<VectorVfsDirectory>();
        parent->AddDirectory(romfs_dir);
        exefs = MakeExeFS(parent);
    }

    Loader::AppLoader_DeconstructedRomDirectory loader{exefs};
    FileSys::VirtualFile romfs;
    REQUIRE(loader.ReadRomFS(romfs) == ResultStatus::Success);
    REQUIRE(romfs != nullptr);
    if (packed != nullptr) {
        REQUIRE(romfs == packed);
    }

    const auto extracted = FileSys::ExtractRomFS(romfs);
    REQUIRE(extracted != nullptr);
    const auto data = extracted->GetFile("data.bin");
    REQUIRE(data != nullptr);
    REQUIRE(data->ReadAllBytes() == payload);

    FileSys::VirtualFile repeated;
    REQUIRE(loader.ReadRomFS(repeated) == ResultStatus::Success);
    REQUIRE(repeated == romfs);
}

TEST_CASE("Directory loader reports absent RomFS", "[core][loader]") {
    const auto exefs = MakeExeFS();
    Loader::AppLoader_DeconstructedRomDirectory loader{exefs};
    FileSys::VirtualFile romfs;
    REQUIRE(loader.ReadRomFS(romfs) == ResultStatus::ErrorNoRomFS);
}

TEST_CASE("Directory loader selects CPU topology without loading a process",
          "[core][loader][topology]") {
    Core::System system;
    for (const size_t prefix_size : {0x200, 0x300}) {
        const auto exefs = MakeExeFS(nullptr, prefix_size);
        const auto main = std::make_shared<VectorVfsFile>(MakeNso(), "main", exefs);
        Loader::AppLoader_DeconstructedRomDirectory from_file{main};
        Loader::AppLoader_DeconstructedRomDirectory from_directory{exefs};
        const auto expected = prefix_size == 0x300 ? Core::Hardware::CpuProfile::Ounce
                                                   : Core::Hardware::CpuProfile::Nx;
        REQUIRE(from_file.GetCpuProfile(system) == expected);
        REQUIRE(from_directory.GetCpuProfile(system) == expected);
        auto selected = Loader::GetLoader(system, main);
        REQUIRE(selected != nullptr);
        REQUIRE(selected->GetCpuProfile(system) == expected);
    }
}

TEST_CASE("Reloading metadata does not retain a previous CPU profile", "[core][loader][topology]") {
    FileSys::ProgramMetadata metadata;
    REQUIRE(metadata.Load(std::make_shared<VectorVfsFile>(MakeMetadata(0x300))) ==
            ResultStatus::Success);
    REQUIRE(metadata.GetCpuProfile() == Core::Hardware::CpuProfile::Ounce);
    REQUIRE(metadata.Reload(std::make_shared<VectorVfsFile>(MakeMetadata(0x200))) ==
            ResultStatus::Success);
    REQUIRE(metadata.GetCpuProfile() == Core::Hardware::CpuProfile::Nx);
    REQUIRE(metadata.Load(std::make_shared<VectorVfsFile>(MakeMetadata(0x300))) ==
            ResultStatus::Success);
    REQUIRE(metadata.Load(nullptr) == ResultStatus::ErrorBadNPDMHeader);
    REQUIRE(metadata.GetCpuProfile() == Core::Hardware::CpuProfile::Nx);
    REQUIRE(FileSys::ProgramMetadata::GetDefault().GetCpuProfile() ==
            Core::Hardware::CpuProfile::Nx);
}
