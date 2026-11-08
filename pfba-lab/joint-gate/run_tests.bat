@echo off
REM CPU tests must use this interpreter only (per task instructions).
cd /d "%~dp0"
"D:\MINICONDA\envs\py312\python.exe" -m unittest tests.test_joint_gate -v %*
