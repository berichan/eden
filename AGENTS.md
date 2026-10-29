# Working on Eden

## Purpose and autonomy

Help make Eden the best emulator it can be: accurate, compatible, fast, stable, usable, and maintainable. The owner welcomes substantial AI assistance, including implementing features, fixing bugs, writing tests, improving documentation, and refactoring systems when the benefit justifies it.

- Carry requested work through investigation, implementation, validation, and a concise handoff. Do not stop at a plan when you can complete the work.
- Make routine technical decisions independently. Address related defects and necessary supporting changes when they improve the result.
- Keep changes coherent and reviewable. Broad permission is not a standing instruction to rewrite unrelated systems or pursue speculative improvements.
- Ask when requirements materially conflict, an important product decision cannot be inferred, or an action could destroy user data. Continue independent work while clarification is pending.

## Policies and local authorization

Read [the coding guidelines](docs/policies/Coding.md) and [the AI policy](docs/policies/AI.md). Consult [the development guidelines](docs/Development.md) and [contribution requirements](CONTRIBUTING.md) as relevant.

For work in this owner's local checkout, this file explicitly authorizes AI implementation, scaffolding, tests, and substantial refactors without prior maintainer discussion. Where the AI policy restricts those local activities, this local authorization takes precedence. It does not waive upstream submission requirements: human review, human testing, understanding of the changes, disclosure, licensing, and human-written submission text remain the human submitter's responsibility.

- Never push commits or open, submit, or merge pull requests on the user's behalf.
- Leave changes uncommitted unless the user explicitly requests a local commit. Do not add `Signed-off-by:`, `Co-authored-by:`, or `Co-developed-by:` tags on behalf of anyone.
- Do not post issues, comments, emails, or other external communications without an explicit user request.
- Preserve license notices and attribution. Follow the repository's SPDX conventions for new source files; do not invent authorship.
- Never expose credentials, console keys, encryption material, private user data, or unpublished security reports in prompts, logs, fixtures, or commits.

## Find the right subsystem

Eden uses C++20 and CMake for its core and desktop frontends, with a separate Android build.

| Area | Starting points |
| --- | --- |
| CPU execution and JIT | `src/core`, `src/dynarmic` |
| Kernel, services, and emulated OS | `src/core/hle`, `docs/HosKernel.md` |
| Graphics and shaders | `src/video_core`, `src/shader_recompiler` |
| Audio, input, and controllers | `src/audio_core`, `src/input_common`, `src/hid_core` |
| Desktop UI and CLI | `src/yuzu`, `src/qt_common`, `src/yuzu_cmd` |
| Android | `src/android`, `docs/build/Android.md` |
| Shared utilities and tests | `src/common`, `src/tests` |
| Build and dependencies | `CMakeLists.txt`, `CMakeModules`, `cpmfile.json`, `externals` |

Use [the design overview](docs/DesignOverview.md) and subsystem documentation to understand existing boundaries before introducing new abstractions. Legacy `yuzu` names still appear throughout the build and code; do not rename them incidentally.

## Implementation standards

- Trace the relevant callers, ownership, state transitions, and error paths before editing. Prefer fixing the cause over hiding symptoms or adding game-specific workarounds.
- Follow surrounding code and the applicable formatting configuration, including `src/.clang-format` for C++. Format touched code without sweeping unrelated files.
- Keep code simple, explicit, and efficient. Reuse existing utilities; introduce dependencies and abstractions only when they solve a concrete problem.
- Respect the project's RTTI-disabled builds and existing platform and architecture guards. Consider supported platforms even when only one can be tested locally.
- Treat guest memory, file formats, IPC data, sizes, and offsets as untrusted. Check bounds, overflow, object lifetimes, synchronization, and failure handling.
- Preserve emulation accuracy and compatibility when optimizing. Measure relevant workloads before claiming speedups; distinguish correctness fixes from deliberate accuracy tradeoffs.
- Keep save data, configuration, and user-facing defaults compatible. Call out any intentional migration or behavior change and provide a safe path where needed.
- Follow existing translation and settings conventions for UI changes. Prefer sensible defaults and automatic detection over unnecessary new settings.
- Keep comments focused on intent, invariants, and non-obvious behavior. Do not narrate obvious code or add/remove comments in unrelated areas.
- Avoid editing vendored dependencies, generated files, translations, or lock data unless the task requires it. Change the source or generator when appropriate.

## Build and validation

Read [the build guide](docs/Build.md), [dependencies](docs/Deps.md), [CMake options](docs/Options.md), and relevant [platform caveats](docs/Caveats.md) before configuring a new build. Reuse a suitable existing build directory and toolchain when available.

For a desktop environment with dependencies installed, the documented starting point is:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build
```

- Build affected targets first, then expand validation according to the scope and risk. Do not disable warnings or tests just to make a failure disappear.
- Core tests use Catch2 in `src/tests`. Enable them with `-DYUZU_TESTS=ON` when configuring, build the `tests` target, then run `ctest --test-dir build --output-on-failure`. For a multi-configuration generator, select the matching build and test configuration.
- Add a regression test for a reproducible bug when practical. Test behavior, boundaries, and failure paths rather than duplicating implementation details.
- Use relevant runtime checks for changes involving rendering, audio, timing, input, or UI; a successful compile alone does not establish correctness. Document checks requiring hardware or game content you cannot access.
- For performance changes, record the workload, build configuration, and before/after measurements. Avoid generalizing from one game, driver, or device.
- For documentation-only changes, check accuracy, referenced paths, and the diff; an emulator build is unnecessary.
- Report exactly what ran and what passed. If dependencies, hardware, or tooling prevent validation, state the limitation and the remaining checks without claiming success.

## Working tree and handoff

- Inspect the working tree before editing. Preserve existing user changes and avoid overwriting or reverting work you did not create.
- Do not use destructive resets, broad cleanup, history rewriting, or deletion of user content to get a clean result.
- Keep build products and temporary diagnostics out of source changes. Review the final diff for accidental edits, formatting churn, and sensitive data.
- Finish with what changed, why it matters, validation results, and any material limitations or decisions the user needs to review. Keep the explanation proportional to the work.
