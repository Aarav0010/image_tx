@echo off
cd /d "%~dp0"
set "PY=python"
if exist "%USERPROFILE%\radioconda\python.exe" set "PY=%USERPROFILE%\radioconda\python.exe"
if exist "C:\radioconda\python.exe" set "PY=C:\radioconda\python.exe"
if exist "D:\BTP\radioconda\python.exe" set "PY=D:\BTP\radioconda\python.exe"
if not "%RADIOCONDA_PY%"=="" set "PY=%RADIOCONDA_PY%"
echo IMAGE TRANSFER - SIMULATION (no hardware needed, about 1 minute)
"%PY%" run_image_transfer.py --sim --tag sim
start "" "%~dp0"
pause
