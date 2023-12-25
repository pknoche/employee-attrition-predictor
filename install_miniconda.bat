@echo off
REM Download Miniconda Installer
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe

REM Install Miniconda in GUI mode
start /wait "" miniconda.exe

REM Delete the Miniconda Installer
del miniconda.exe

echo Miniconda installation completed
