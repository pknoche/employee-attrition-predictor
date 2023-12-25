@echo off
REM Check if the 'env' directory exists, create it if it doesn't
IF NOT EXIST "env" (
    mkdir env
)

REM Create the Conda environment
conda env create -f environment.yml -p ./env

REM Reminder to activate the environment
echo To activate the environment, run: conda activate ./env

REM Instructions for launching application
echo To launch the application, activate the environment and run: voila Employee_Attrition_Predictor.ipynb OR double click the launch_application file located within the code folder
