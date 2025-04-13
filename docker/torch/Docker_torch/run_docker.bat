@echo off
echo [INFO] Building Docker image...
docker build -t noise-pytorch .

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to build Docker image. Check Dockerfile path and try again.
    pause
    exit /b
)

echo [INFO] Running Docker container...
docker run --rm -v %cd%\..:/app --gpus all noise-pytorch

echo [INFO] All tasks completed.
pause
