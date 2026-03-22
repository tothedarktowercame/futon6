@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"
for %%I in ("%REPO_ROOT%\..\..\gh\mfuton") do set "DEFAULT_MFUTON_ROOT=%%~fI"
for %%I in ("%REPO_ROOT%\..\futon3c-mfuton-overlay") do set "DEFAULT_FUTON3C_OVERLAY_ROOT=%%~fI"
for %%I in ("%REPO_ROOT%\..\futon3c") do set "DEFAULT_FUTON3C_ROOT=%%~fI"

if not defined FUTON3C_ROOT (
  if exist "%DEFAULT_FUTON3C_OVERLAY_ROOT%\scripts\windows\futon-windows.bat" (
    set "FUTON3C_ROOT=%DEFAULT_FUTON3C_OVERLAY_ROOT%"
  ) else (
    set "FUTON3C_ROOT=%DEFAULT_FUTON3C_ROOT%"
  )
)

set "FUTON3C_LAUNCHER=%FUTON3C_ROOT%\scripts\windows\futon-windows.bat"
if not exist "%FUTON3C_LAUNCHER%" (
  1>&2 echo [frontiermath-local] ERROR: missing futon3c launcher %FUTON3C_LAUNCHER%.
  1>&2 echo [frontiermath-local] Set FUTON3C_ROOT to a futon3c checkout and retry.
  exit /b 1
)

set "FORWARD_ARGS="
:parse_args
if "%~1"=="" goto args_parsed
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--remote-irc" (
  1>&2 echo [frontiermath-local] ERROR: --remote-irc is incompatible with this local FrontierMath wrapper.
  exit /b 1
)
if defined FORWARD_ARGS (
  set "FORWARD_ARGS=!FORWARD_ARGS! %1"
) else (
  set "FORWARD_ARGS=%1"
)
shift
goto parse_args
:args_parsed

if not defined BRIDGE_BOTS set "BRIDGE_BOTS=codex"
if not defined FUTON3C_CODEX_AGENT_ID set "FUTON3C_CODEX_AGENT_ID=codex-1"
if not defined FUTON3C_REGISTER_CLAUDE set "FUTON3C_REGISTER_CLAUDE=false"
if not defined FUTON3C_RELAY_CLAUDE set "FUTON3C_RELAY_CLAUDE=false"
if not defined IRC_CHANNEL set "IRC_CHANNEL=#futon"
if not defined IRC_COMMAND_OWNER_AGENT_MAP set "IRC_COMMAND_OWNER_AGENT_MAP=#futon:codex-1,#math:codex-1"
if not defined CODEX_SESSION_FILE set "CODEX_SESSION_FILE=%REPO_ROOT%\.state\codex-frontiermath-local\session-id"
if not defined CODEX_CWD set "CODEX_CWD=%REPO_ROOT%"
if not defined CODEX_BRIDGE_SUMMARY_MODE set "CODEX_BRIDGE_SUMMARY_MODE=raw"
if not defined MFUTON_ROOT set "MFUTON_ROOT=%DEFAULT_MFUTON_ROOT%"
if not defined FUTON3C_PROOF_STATE_ROOT set "FUTON3C_PROOF_STATE_ROOT=%MFUTON_ROOT%\data\frontiermath-local\FM-001\active"
if not defined FUTON1A_PORT set "FUTON1A_PORT=7271"
if not defined FUTON3C_PORT set "FUTON3C_PORT=7270"
if not defined FUTON3C_DRAWBRIDGE_PORT set "FUTON3C_DRAWBRIDGE_PORT=7768"
if not defined FUTON3C_IRC_PORT set "FUTON3C_IRC_PORT=7667"
if not defined BRIDGE_HTTP_PORT set "BRIDGE_HTTP_PORT=7769"
if not defined INVOKE_BASE set "INVOKE_BASE=http://127.0.0.1:%FUTON3C_PORT%"
if not defined XDG_RUNTIME_DIR set "XDG_RUNTIME_DIR=%MFUTON_ROOT%\data\tmp\frontiermath-local\runtime"

if not exist "%XDG_RUNTIME_DIR%" (
  mkdir "%XDG_RUNTIME_DIR%" >nul 2>nul
  if errorlevel 1 (
    1>&2 echo [frontiermath-local] ERROR: unable to create runtime dir %XDG_RUNTIME_DIR%.
    exit /b 1
  )
)

if not exist "%FUTON3C_PROOF_STATE_ROOT%" (
  1>&2 echo [frontiermath-local] ERROR: missing FrontierMath proof-state root %FUTON3C_PROOF_STATE_ROOT%.
  1>&2 echo [frontiermath-local] Set FUTON3C_PROOF_STATE_ROOT directly or set MFUTON_ROOT to an mfuton checkout.
  exit /b 1
)

echo [frontiermath-local] futon6-owned FrontierMath local lane
echo [frontiermath-local] futon6=%REPO_ROOT%
echo [frontiermath-local] futon3c=%FUTON3C_ROOT%
echo [frontiermath-local] mfuton=%MFUTON_ROOT%
echo [frontiermath-local] session=%CODEX_SESSION_FILE%
echo [frontiermath-local] codex cwd=%CODEX_CWD%
echo [frontiermath-local] proof-state-root=%FUTON3C_PROOF_STATE_ROOT%
echo [frontiermath-local] ports=http:%FUTON3C_PORT% health:%FUTON1A_PORT% irc:%FUTON3C_IRC_PORT% drawbridge:%FUTON3C_DRAWBRIDGE_PORT%
echo [frontiermath-local] bridge-http=%BRIDGE_HTTP_PORT%
echo [frontiermath-local] invoke-base=%INVOKE_BASE%
echo [frontiermath-local] runtime-dir=%XDG_RUNTIME_DIR%
echo [frontiermath-local] primary channel=%IRC_CHANNEL%
echo [frontiermath-local] extra channel=#math
if defined IRC_COMMAND_OWNER_AGENT_MAP echo [frontiermath-local] owner map=%IRC_COMMAND_OWNER_AGENT_MAP%
echo [frontiermath-local] NOTE: this Windows wrapper binds the current mfuton-backed FM-001 active root without changing futon3c's generic proof tooling surface.
echo [frontiermath-local] NOTE: default CODEX_CWD keeps FrontierMath work rooted in futon6 instead of scattering into whichever repo launched the runtime.

call "%FUTON3C_LAUNCHER%" dev --math-irc %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:usage
echo Usage: scripts\frontiermath\local-futon3c-windows.bat [futon3c-dev-flags]
echo.
echo Starts a local FrontierMath-oriented futon3c dev lane owned by futon6.
echo.
echo Environment overrides:
echo   FUTON3C_ROOT                 path to futon3c checkout ^(default: sibling futon3c-mfuton-overlay when present, else sibling futon3c^)
echo   MFUTON_ROOT                  path to mfuton checkout used for the local FM-001 active root
echo   CODEX_SESSION_FILE           continuity file for the codex lane
echo   CODEX_CWD                    working directory for codex execution ^(default: futon6 root^)
echo   FUTON3C_PROOF_STATE_ROOT     explicit proof-state root override ^(default: %%MFUTON_ROOT%%\data\frontiermath-local\FM-001\active^)
echo   FUTON1A_PORT                 local health/evidence port ^(default 7271^)
echo   FUTON3C_PORT                 local futon3c HTTP port ^(default 7270^)
echo   FUTON3C_DRAWBRIDGE_PORT      local drawbridge eval port ^(default 7768^)
echo   FUTON3C_IRC_PORT             local IRC port ^(default 7667^)
echo   BRIDGE_HTTP_PORT             local ngircd bridge /say port ^(default 7769^)
echo   INVOKE_BASE                  bridge target futon3c base ^(default http://127.0.0.1:%%FUTON3C_PORT%%^)
echo   XDG_RUNTIME_DIR              local runtime dir for bridge pid/health files
echo   IRC_CHANNEL                  primary IRC room ^(default #futon^)
echo   IRC_COMMAND_OWNER_AGENT_MAP  optional room-owner map for bare ! commands
echo.
echo Notes:
echo   - always adds #math via futon3c's supported --math-irc lane
echo   - rejects --remote-irc; this wrapper is local-only
echo   - defaults to an isolated local rehearsal port quartet ^(7271/7270/7667/7768^)
echo   - isolates bridge pid/health files and /say HTTP to the FrontierMath local lane
echo   - keeps the futon3c proof tooling surface generic and only binds a local owner-side default
exit /b 0
