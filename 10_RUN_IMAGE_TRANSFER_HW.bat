@echo off
cd /d "%~dp0"
REM Finds the radioconda Python (or set RADIOCONDA_PY to your python.exe)
set "PY=python"
if exist "%USERPROFILE%\radioconda\python.exe" set "PY=%USERPROFILE%\radioconda\python.exe"
if exist "C:\radioconda\python.exe" set "PY=C:\radioconda\python.exe"
if exist "D:\BTP\radioconda\python.exe" set "PY=D:\BTP\radioconda\python.exe"
if not "%RADIOCONDA_PY%"=="" set "PY=%RADIOCONDA_PY%"
echo IMAGE TRANSFER over the B210 - 10 MHz bandwidth (BPSK + QPSK)
echo B210 connected: TX antenna on RF0 TX1 RX1, RX antenna on RF1 RX2.
echo Close uhd_fft / GNU Radio Companion first.
echo.
set /p COND=Name this condition (e.g. clear, barrier): 
set RXG=
set /p RXG=RX gain in dB (just press Enter for 40): 
if "%RXG%"=="" set RXG=40
set GAINS=
set /p GAINS=TX gains (just press Enter for 20,22,24,26,28,30): 
if "%GAINS%"=="" set GAINS=20,22,24,26,28,30
"%PY%" run_image_transfer.py --tag "%COND%" --rx-gain %RXG% --gains %GAINS%
start "" "%~dp0"
pause
