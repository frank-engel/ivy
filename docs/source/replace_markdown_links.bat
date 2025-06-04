@echo off
setlocal enabledelayedexpansion

set "list_file=list_of_links.txt"

for /f "tokens=1,* delims=]" %%a in (%list_file%) do (
    set "link=%%b"
    set "link=!link:[=!"
    for %%f in (*.md) do (
        call :replace "%%a" "!link!" "%%~f"
    )
)

exit /b

:replace
set "find=%~1"
set "replace=%~2"
set "file=%~3"
set "tempfile=%file%.temp"

(for /f "delims=" %%l in ('type "%file%" ^& break ^> "%tempfile%" ') do (
    set "line=%%l"
    setlocal enabledelayedexpansion
    set "line=!line:%find%=!"
    if not "!line!"=="%%l" (
        set "line=!line:[=!"
        set "line=!line:)=!"
        echo ^<a href=!replace!^>!line!^</a^>
    ) else (
        echo(!line!
    )
    endlocal
)) >nul

move /y "%tempfile%" "%file%" >nul