# M2 Fast Backend Build

This document records only the reproducible build steps for the M2 pybind11 C++ fast backend.

## Dependencies

- Python 3.11+ for local Windows development; CI currently uses Python 3.12.
- CMake.
- pybind11 3.1.0.
- NumPy and pytest for the test suite.
- PyTorch 2.9.1 for the complete M2 CPU test suite.

Install pybind11 into the Python environment used to run the tests:

    python -m pip install pybind11==3.1.0

## Windows MSVC requirement

The supported Windows compiler is 64-bit MSVC. Install the Visual Studio C++ workload so that vswhere can resolve Microsoft.VisualStudio.Component.VC.Tools.x86.x64, then run the build from a shell initialized by VsDevCmd.bat.

The validated local toolchain was:

- MSVC compiler 19.51.36248 x64.
- Windows SDK 10.0.26100.0.
- CMake.
- Python 3.11.9.
- pybind11 3.1.0.

## CMake configure

From the repository root in the same Python environment used for pytest:

    $pybind = python -m pybind11 --cmakedir
    cmake -S cpp/m2_fast_backend -B build/m2_fast_backend `
      -G "NMake Makefiles" `
      -DCMAKE_BUILD_TYPE=Release `
      -DPython_EXECUTABLE="$(Get-Command python | Select-Object -ExpandProperty Source)" `
      -Dpybind11_DIR="$pybind"

If using a multi-config Visual Studio generator, omit the NMake generator and select Release at build time.

## CMake build

NMake single-config build:

    cmake --build build/m2_fast_backend

Multi-config build:

    cmake --build build/m2_fast_backend --config Release

## Extension output

The compiled module is written under:

    build/m2_fast_backend/

The exact filename is platform/Python specific. On the validated Windows Python 3.11 environment it is:

    _m2_fast_backend.cp311-win_amd64.pyd

src/game2048/m2_fast_backend.py locates the extension in that build directory.

## Run pytest

After the extension is built:

    python -m pytest

The M2 candidate requires the full suite to run with no skipped or xfailed backend tests.
