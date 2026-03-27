@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

if not defined MFUTON_HOME (
  1>&2 echo [frontiermath-local] ERROR: MFUTON_HOME is required.
  1>&2 echo [frontiermath-local] Set MFUTON_HOME to the mfuton checkout that owns the Windows control logic for this wrapper.
  exit /b 1
)

for %%I in ("%MFUTON_HOME%") do set "MFUTON_HOME=%%~fI"
set "MFUTON_CODEX_PY=%MFUTON_HOME%\agent_skills\development\codex-python.bat"
set "MFUTON_LAUNCHER=%MFUTON_HOME%\src\mfuton\development\frontiermath_local_futon3c_windows.py"
set "FRONTIERMATH_LOCAL_CONFIG=%REPO_ROOT%\config\frontiermath-local-futon3c-windows.json"

if not exist "%MFUTON_CODEX_PY%" (
  1>&2 echo [frontiermath-local] ERROR: missing %MFUTON_CODEX_PY%.
  exit /b 1
)

if not exist "%MFUTON_LAUNCHER%" (
  1>&2 echo [frontiermath-local] ERROR: missing %MFUTON_LAUNCHER%.
  exit /b 1
)

if not exist "%FRONTIERMATH_LOCAL_CONFIG%" (
  1>&2 echo [frontiermath-local] ERROR: missing %FRONTIERMATH_LOCAL_CONFIG%.
  exit /b 1
)

call "%MFUTON_CODEX_PY%" "%MFUTON_LAUNCHER%" --futon6-root "%REPO_ROOT%" --config "%FRONTIERMATH_LOCAL_CONFIG%" %*
exit /b %ERRORLEVEL%
