@echo off
REM Activate the Conda environment
call conda activate ../env

REM Launching Voila with the specified notebook
call voila Employee_Attrition_Predictor.ipynb
