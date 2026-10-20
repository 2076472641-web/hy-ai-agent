@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=C:\Users\坚果520\AppData\Roaming\Python\Python312\site-packages;%PYTHONPATH%
echo ============================================
echo   docx 批量转 markdown
echo   当前目录: %cd%
echo ============================================
echo.
"D:\software\Anaconda\python.exe" convert_docx2md.py
echo.
pause
