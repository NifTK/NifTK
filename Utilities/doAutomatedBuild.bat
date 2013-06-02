@echo ***** NifTK Automated Build Script - v.17 *****
@echo. 

@REM ***** Attempt to enable Command extensions *****
@setlocal enableextensions 

@REM *****  Setting localised variables - Change these to match your system!!  *****
@set "VS_LOCATION=c:\Program Files (x86)\Microsoft Visual Studio 10.0"
@set "CMAKE_LOCATION=c:\Program Files\CMake_2.8.9-git\bin"
@set "BUILD_LOCATION=d:\a"
@set "PUTTY_LOCATION=c:\Program Files (x86)\PuTTY\"
@set "OPENSSL_LOCATION=c:\OpenSSL-Win64\bin\"

@REM *****  Set your build type 64bit/32bit  *****
@set "BTYPE=x64"
@REM @set "BTYPE=Win32"

@REM *****  Set your Visual Studio Version  *****
@set "VSVER=devenv.exe"
@REM @set "VSVER=VCExpress.exe"

@REM *****  Set your CMake generator - the tool you're going to use to build NifTK  *****
@set "CMAKE_GENERATOR=Visual Studio 10 Win64"

@REM ***** Possible options are the following: ***** 
@REM NMake Makefiles             = Generates NMake makefiles.
@REM NMake Makefiles JOM         = Generates JOM makefiles.
@REM Visual Studio 10            = Generates Visual Studio 10 project files.
@REM Visual Studio 10 Win64      = Generates Visual Studio 10 Win64 project files.
@REM Visual Studio 11            = Generates Visual Studio 11 project files.
@REM Visual Studio 11 Win64      = Generates Visual Studio 11 Win64 project files.
@REM Visual Studio 9 2008        = Generates Visual Studio 9 2008 project files
@REM Visual Studio 9 2008 Win64  = Generates Visual Studio 9 2008 Win64 project files.

@REM *****  Configuring the date-stamp  *****
@set DATESTAMP=%date%
@set DATESTAMP=%DATESTAMP: =-%
@set DATESTAMP=%DATESTAMP::=%
@set DATESTAMP=%DATESTAMP:/=-%
@set DATESTAMP=%DATESTAMP:.=-%

@echo Visual Studio location: %VS_LOCATION%
@echo Visual Studio version:  %VSVER%
@echo CMake location:         %CMAKE_LOCATION%
@echo Build folder:           %BUILD_LOCATION%
@echo CMake generator:        %CMAKE_GENERATOR%
@echo Date Stamp:             %DATESTAMP%
@echo.

@rem stop visual studio recycling already running instances of msbuild.exe. we want clean ones.
@rem http://stackoverflow.com/questions/12174877/visual-studio-2012-rtm-has-msbuild-exe-in-memory-after-close
@set MSBUILDDISABLENODEREUSE=1

@if ["%BTYPE%"] == ["x64"] (
  @REM *****  Setting environmental variables for x64  *****
  @if ["%CMAKE_GENERATOR%"]==["Visual Studio 9 2008 Win64"] (
    @call "%VS_LOCATION%\VC\bin\amd64\vcvarsamd64.bat"
  ) else (
    @call "%VS_LOCATION%\VC\bin\amd64\vcvars64.bat"
  )
) else (
  @REM *****  Setting environmental variables for Win32  *****
  @call "%VS_LOCATION%\VC\bin\vcvars32.bat"
)

@REM *****  Configure which branch to build - defaults to dev *****
@if [%1]==[] (
  @set BRANCH=dev
) else (
  @set BRANCH=%1
)

@REM *****  Configure which config to build - defaults to release *****
@if [%2]==[] (
  @set BCONF=Release
) else (
  @set BCONF=%2
)

@REM *****  Sanity check the build config parameter *****
@if /I NOT [%BCONF%]==[Release] if /I NOT [%BCONF%]==[Debug] (
    @echo Build config incorrectly set to %BCONF%, it has to be either 'Debug' or 'Release' - defaulting to 'Release'
	@set BCONF=Release
    )
	
@REM *****  Configure the current build path  *****

@REM First shorten the branch name and buildconf
@set BR=%BRANCH%
@set BR=%BR:~0,3%

@set BC=%BCONF%
@set BC=%BC:~0,3%

@set "BUILDPATH=%BUILD_LOCATION%\%BR%-%BC%-%DATESTAMP%"
@REM @echo %BUILDPATH%

@REM *****  Configure the current VS Config Script  *****
@set "VSCONFSTRING=%BCONF%^|%BTYPE%"	
	
@REM *****  Disable GIT ssl verification  *****
@set GIT_SSL_NO_VERIFY=1

@echo.
@REM *****  Clean off the local NIFTK repo  *****
@echo Cleaning the local NifTK repository....
@if exist "%BUILD_LOCATION%\NifTK" rd /s /q "%BUILD_LOCATION%\NifTK"

@REM *****  Cleaning off the local build folder  *****
@echo Cleaning the previous builds....
@REM @echo Build Location: %BUILD_LOCATION%
@FOR /D %%X in ("%BUILD_LOCATION%\%BR%-%BC%-*") DO @(
  @rd /s /q %%X
  @if exist [%%X] rd /s /q %%X
)

@REM pause
@REM *****  Create new local build folder  *****
@md %BUILDPATH%

@REM *****  Clear buildlogs  *****
@echo Cleaning previous build logs....
@cd %BUILD_LOCATION%
@del log_*

@REM pause
@echo.
@echo *****  Fetch the latest source from GIT  *****
call git clone https://cmicdev.cs.ucl.ac.uk/git/NifTK
@cd "%BUILD_LOCATION%\NifTK"
@call git checkout %BRANCH%
@call git pull origin %BRANCH%

@echo.
@echo *****  Configuring the build with CMake  *****
@echo Building %BRANCH%-%BCONF%-%BTYPE%
@echo To: %BUILDPATH%
@echo Current VS Config string is: %VSCONFSTRING%
@echo.
@REM pause

@REM *****  Run CMAKE  *****
@echo Running CMake....
@cd %BUILDPATH%
@call "%CMAKE_LOCATION%\cmake.exe" -DCMAKE_BUILD_TYPE=%BCONF% -DNIFTK_BUILD_ALL_APPS=ON -DNIFTK_USE_GIT_PROTOCOL=ON -DBUILD_TESTING=ON -DBUILD_COMMAND_LINE_PROGRAMS=ON -DBUILD_COMMAND_LINE_SCRIPTS=ON -DNIFTK_GENERATE_DOXYGEN_HELP=ON -G "%CMAKE_GENERATOR%" "%BUILD_LOCATION%\NIFTK" >"%BUILD_LOCATION%\log_cmake.txt"
@echo. 

@REM pause

@REM *****  Run Visual Studio to build the current build conf  *****
@echo Running VS....
"%VS_LOCATION%\Common7\IDE\%VSVER%" /build %BCONF% /project ALL_BUILD /projectconfig %VSCONFSTRING% /out %BUILD_LOCATION%\log_build-%BRANCH%-%BCONF%.txt %BUILDPATH%\NIFTK-SUPERBUILD.sln
  
@echo. 
@REM pause  

@REM  *****  Set PATH and Environment for NifTK  *****
@cd "%BUILDPATH%\NIFTK-build\"
@set CL=/D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_DEPRECATE
@set LINK=/LARGEADDRESSAWARE
@set NIFTK_DRC_ANALYZE=ON

@set "BATFILEPATH=%BUILDPATH%\NIFTK-build\bin\startNiftyView_%BCONF%.bat"
@set PATHSTRING=" "

@echo Reading %BATFILEPATH%....
@echo.

@set /a counter=1
@setlocal ENABLEDELAYEDEXPANSION
@for /f ^"usebackq^ eol^=^

^ delims^=^" %%a in (%BATFILEPATH%) do @(
        @if "!counter!"==4 goto :eof
		@SET var!counter!=%%a
	    @set /a counter+=1
)

@set PATHSTRING=%var1:/=\%
@set PATHSTRING=%PATHSTRING:PATH=%
@set PATHSTRING=%PATHSTRING:~1,-2%
@PATH=%PATHSTRING%;%SystemRoot%;%SystemRoot%\system32;%SystemRoot%\System32\Wbem;%OPENSSL_LOCATION%;%BUILDPATH%\curl-build\lib;%BUILDPATH%\curl-build\lib\%BCONF%

@if defined CUDA_PATH PATH=%PATH%;%CUDA_PATH%\bin

@echo.
@echo The current system path:
@echo %PATH%
@echo.

@REM *****  Run CTEST  *****
@echo Running CTest....
"%CMAKE_LOCATION%\ctest.exe" -C Release -E CTE-Stream -D NightlyStart >%BUILD_LOCATION%\log_ctest.txt
"%CMAKE_LOCATION%\ctest.exe" -C Release -E CTE-Stream -D NightlyConfigure >>%BUILD_LOCATION%\log_ctest.txt
"%CMAKE_LOCATION%\ctest.exe" -C Release -E CTE-Stream -D NightlyBuild >>%BUILD_LOCATION%\log_ctest.txt
"%CMAKE_LOCATION%\ctest.exe" -C Release -E CTE-Stream -D NightlyTest >>%BUILD_LOCATION%\log_ctest.txt
"%CMAKE_LOCATION%\ctest.exe" -C Release -E CTE-Stream -D NightlySubmit >>%BUILD_LOCATION%\log_ctest.txt
@echo.

@REM *****  Package the installer *****
@echo Packaging....
"%VS_LOCATION%\Common7\IDE\%VSVER%" /build %BCONF% /project PACKAGE /projectconfig %VSCONFSTRING% /out "%BUILD_LOCATION%\log_build-package.txt" NIFTK.sln
@echo.

@echo Uploading package to server....
CALL "%BUILD_LOCATION%\copy_exe_nightly.bat"

@echo.
@echo ***** NifTK Automated Build Script FINISHED *****
@endlocal
@endlocal