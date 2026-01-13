@echo off
echo Setting up data directory structure...

set DATA_DIR=backend\data

REM Create directories
mkdir "%DATA_DIR%\submissions" 2>nul
mkdir "%DATA_DIR%\evidence\2026\01" 2>nul
mkdir "%DATA_DIR%\evidence\2026\02" 2>nul
mkdir "%DATA_DIR%\evidence\2026\03" 2>nul
mkdir "%DATA_DIR%\evidence\2026\04" 2>nul
mkdir "%DATA_DIR%\evidence\2026\05" 2>nul
mkdir "%DATA_DIR%\evidence\2026\06" 2>nul
mkdir "%DATA_DIR%\evidence\2026\07" 2>nul
mkdir "%DATA_DIR%\evidence\2026\08" 2>nul
mkdir "%DATA_DIR%\evidence\2026\09" 2>nul
mkdir "%DATA_DIR%\evidence\2026\10" 2>nul
mkdir "%DATA_DIR%\evidence\2026\11" 2>nul
mkdir "%DATA_DIR%\evidence\2026\12" 2>nul
mkdir "%DATA_DIR%\reports" 2>nul
mkdir "%DATA_DIR%\cache" 2>nul

REM Create .gitkeep files
type nul > "%DATA_DIR%\.gitkeep"
type nul > "%DATA_DIR%\submissions\.gitkeep"
type nul > "%DATA_DIR%\evidence\.gitkeep"
for /L %%m in (1,1,12) do (
    if %%m LSS 10 (
        type nul > "%DATA_DIR%\evidence\2026\0%%m\.gitkeep"
    ) else (
        type nul > "%DATA_DIR%\evidence\2026\%%m\.gitkeep"
    )
)
type nul > "%DATA_DIR%\reports\.gitkeep"
type nul > "%DATA_DIR%\cache\.gitkeep"

echo Data directory structure created successfully!
