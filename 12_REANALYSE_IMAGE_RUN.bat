@echo off
cd /d "%~dp0"
set "PY=python"
if exist "%USERPROFILE%\radioconda\python.exe" set "PY=%USERPROFILE%\radioconda\python.exe"
if exist "C:\radioconda\python.exe" set "PY=C:\radioconda\python.exe"
if exist "D:\BTP\radioconda\python.exe" set "PY=D:\BTP\radioconda\python.exe"
if not "%RADIOCONDA_PY%"=="" set "PY=%RADIOCONDA_PY%"
echo Re-decode a saved run (its captures.npz) with the current receiver. No hardware needed.
set /p RUN=Folder name (e.g. results_image_20261102_0149_10MHz_clear): 
"%PY%" reanalyse_image_run.py "%RUN%"
start "" "%~dp0%RUN%\reanalysed"
pause
