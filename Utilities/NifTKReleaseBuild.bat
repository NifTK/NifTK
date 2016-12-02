@echo off

rem The script is invoked from the gitlab-ci.yml job without arguments.
rem The script assumes that the following environment variables are set:
rem
rem   - src_dir: points to the NifTK source folder
rem   - ep_dir: points to the folder where the external projects are (to be) built
rem   - sb_dir: points to the folder where NifTK is to be built (superbuild folder)
rem
rem The script also assumes that the current directory is %sb_dir%
rem
rem The script builds NifTK in release mode with testing on.

rem Enable command extensions to allow checking exit status of commands.
setlocal EnableExtensions 

set "VS_DIR=c:/Program Files (x86)/Microsoft Visual Studio 11.0"
set "CMAKE_DIR=c:/Program Files/CMake"
set "OPENSSL_DIR=c:/OpenSSL-Win64"
set "QT_DIR=c:/Qt/4.8.7"
set "QTDIR=%QT_DIR%"

set GIT_SSL_NO_VERIFY=1

set "BTYPE=x64"
rem set "BTYPE=Win32"

set "VS_COMMAND=devenv.com"
rem set "VS_COMMAND=VCExpress.exe"

set "CMAKE_GENERATOR=Visual Studio 11 Win64"

rem Possible options are the following:
rem NMake Makefiles             = Generates NMake makefiles.
rem NMake Makefiles JOM         = Generates JOM makefiles.
rem Visual Studio 11            = Generates Visual Studio 11 project files. Corresponding to Visual Studio 2012.
rem Visual Studio 11 Win64      = Generates Visual Studio 11 Win64 project files. Corresponding to Visual Studio 2012.
rem Visual Studio 12            = Generates Visual Studio 12 project files. Corresponding to Visual Studio 2013.
rem Visual Studio 12 Win64      = Generates Visual Studio 12 Win64 project files. Corresponding to Visual Studio 2013.

echo Visual Studio folder:   %VS_DIR%
echo Visual Studio command:  %VS_COMMAND%
echo CMake folder:           %CMAKE_DIR%
echo Source folder:          %src_dir%
echo Build folder:           %sb_dir%
echo CMake generator:        %CMAKE_GENERATOR%

rem stop visual studio recycling already running instances of msbuild.exe. we want clean ones.
rem http://stackoverflow.com/questions/12174877/visual-studio-2012-rtm-has-msbuild-exe-in-memory-after-close
set MSBUILDDISABLENODEREUSE=1

if ["%BTYPE%"] == ["x64"] (
  call "%VS_DIR%/VC/bin/amd64/vcvars64.bat"
) else (
  call "%VS_DIR%/VC/bin/vcvars32.bat"
)

set BCONF=Release

set "VS_CONF=%BCONF%^|%BTYPE%"
echo Visual Studio config:   %VS_CONF%

set CL=/D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_DEPRECATE
set LINK=/LARGEADDRESSAWARE

rem The git usr/bin directory is needed for the 'tee' command.
set "PATH=%CMAKE_DIR%/bin;c:/Program Files/Git/usr/bin;%QT_DIR%/bin;%OPENSSL_DIR%/bin;%VS_DIR%/Common7/IDE;%PATH%"
echo PATH:                   %PATH%

call cmake.exe ^
    -DCMAKE_BUILD_TYPE:STRING=%BCONF% ^
    -DDESIRED_QT_VERSION:STRING=4 ^
    -DOPENCV_WITH_FFMPEG:BOOL=OFF ^
    -DNIFTK_USE_CUDA:BOOL=OFF ^
    -DBUILD_TESTING:BOOL=OFF ^
    -DBUILD_COMMAND_LINE_PROGRAMS:BOOL=ON ^
    -DBUILD_COMMAND_LINE_SCRIPTS:BOOL=ON ^
    -DNIFTK_GENERATE_DOXYGEN_HELP:BOOL=ON ^
    -DBUILD_Python:BOOL=ON ^
    -DBUILD_CAFFE:BOOL=ON ^
    -DNIFTK_Apps/NiftyView:BOOL=ON ^
    -DNIFTK_Apps/NiftyIGI:BOOL=ON ^
    -DNIFTK_Apps/NiftyMIDAS:BOOL=ON ^
    -G "%CMAKE_GENERATOR%" "%src_dir%"
if %ERRORLEVEL% NEQ 0 exit /B 1

%VS_COMMAND% /build %BCONF% /project ALL_BUILD /projectconfig %VS_CONF% %sb_dir%/NIFTK-superbuild.sln | tee %sb_dir%/build.log 2>&1
if %ERRORLEVEL% NEQ 0 exit /B 2
