@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PYTHON_BIN=%PYTHON_BIN%"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

"%PYTHON_BIN%" "%SCRIPT_DIR%\advance-proof-cycle-from-local-run.py" %*
exit /b %ERRORLEVEL%
