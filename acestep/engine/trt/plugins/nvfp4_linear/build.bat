@echo off
REM Build NVFP4 Linear plugin DLL for TRT 10.16 on Windows (Blackwell sm_120).
REM Requires: MSVC BuildTools 14.x, CUDA 12.8, TRT 10.16 headers (vendored).

setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo Failed to set up MSVC env via vcvars64.bat
  exit /b 1
)

set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..\..\..\..\..
set TRT_INC=%SCRIPT_DIR%..\_trt_headers\include
set TRT_LIB=%SCRIPT_DIR%..\_build
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_LIB=%CUDA_PATH%\lib\x64

if not exist "%TRT_INC%\NvInferRuntime.h" (
  echo TRT headers not found at %TRT_INC%
  exit /b 2
)
if not exist "%TRT_LIB%\nvinfer_10.lib" (
  echo nvinfer_10.lib not found at %TRT_LIB%
  echo Run plugins\_build\ steps to generate the import lib from nvinfer_10.dll
  exit /b 3
)

"%CUDA_PATH%\bin\nvcc.exe" ^
  -O2 ^
  -std=c++17 ^
  -arch=sm_120 ^
  -shared ^
  --use_fast_math ^
  --expt-relaxed-constexpr ^
  -DNVFP4_PLUGIN_BUILD ^
  -I "%TRT_INC%" ^
  -L "%CUDA_LIB%" ^
  -L "%TRT_LIB%" ^
  -lcublasLt ^
  -lcudart ^
  -lnvinfer_10 ^
  -o "%SCRIPT_DIR%nvfp4_linear_plugin.dll" ^
  "%SCRIPT_DIR%nvfp4_linear_plugin.cu"

if errorlevel 1 (
  echo Build FAILED
  exit /b 4
)

echo Build OK: %SCRIPT_DIR%nvfp4_linear_plugin.dll
endlocal
