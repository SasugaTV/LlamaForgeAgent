@echo off
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Starting LlamaForge Agent...
python main.py
pause