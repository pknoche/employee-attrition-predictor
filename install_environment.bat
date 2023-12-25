@echo off
REM Check if the 'env' directory exists, create it if it doesn't
IF NOT EXIST "env" (
    mkdir env
)

REM Create the Conda environment
call conda env create -f environment.yml -p ./env

echo Environment installation completed
pause