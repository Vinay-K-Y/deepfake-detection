@echo off
REM DeepTrust Project - Quick Start Script for Windows
REM Save this as: run_deeptrust.bat

echo ======================================================================
echo DEEP TRUST - DEEPFAKE DETECTION PROJECT
echo ======================================================================
echo.

echo What would you like to do?
echo.
echo 1. Run API only (Backend)
echo 2. Run Full Web App (API + Frontend)
echo 3. Evaluate Model (Get 99.74%% results)
echo 4. Test Setup (Verify everything is installed)
echo 5. Install Requirements
echo 6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto run_api
if "%choice%"=="2" goto run_webapp
if "%choice%"=="3" goto evaluate
if "%choice%"=="4" goto test_setup
if "%choice%"=="5" goto install
if "%choice%"=="6" goto end

:run_api
echo.
echo Starting FastAPI backend...
echo API will be available at: http://localhost:8000
echo API docs available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
python deeptrust_api.py
goto end

:run_webapp
echo.
echo Starting DeepTrust Web Application...
echo.
echo Step 1: Starting API backend...
start cmd /k "python deeptrust_api.py"
timeout /t 3 /nobreak > nul

echo Step 2: Starting frontend server...
start cmd /k "python -m http.server 3000"
timeout /t 2 /nobreak > nul

echo.
echo ✓ Application started!
echo.
echo Open your browser and go to:
echo http://localhost:3000/index.html
echo.
echo Press any key to stop both servers...
pause > nul
taskkill /F /FI "WindowTitle eq *deeptrust_api.py*" > nul 2>&1
taskkill /F /FI "WindowTitle eq *http.server*" > nul 2>&1
goto end

:evaluate
echo.
echo Running model evaluation...
echo This will generate:
echo   - confusion_matrix.png
echo   - roc_curve.png
echo   - Performance metrics
echo.
python evaluate.py
echo.
echo Press any key to continue...
pause > nul
goto end

:test_setup
echo.
echo Verifying project setup...
echo.
python test_setup.py
echo.
echo Press any key to continue...
pause > nul
goto end

:install
echo.
echo Installing required packages...
echo This may take a few minutes...
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart
pip install mediapipe opencv-python pillow
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
echo.
echo ✓ Installation complete!
echo.
echo Press any key to continue...
pause > nul
goto end

:end
echo.
echo Thank you for using DeepTrust!
echo.
