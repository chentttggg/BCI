@echo off
cd /d %~dp0\..
.venv311\Scripts\pyinstaller.exe --noconfirm --clean packaging\guess_number_researcher.spec
echo.
echo Built: dist\GuessNumberResearcher.exe
pause
