@rem The script is invoked from the gitlab-ci.yml job without arguments.
@rem The script assumes that the following environment variables are set:
@rem
@rem   - src_dir: points to the NifTK source folder
@rem   - ep_dir: points to the folder where the external projects are (to be) built
@rem   - sb_dir: points to the folder where NifTK is to be built (superbuild folder)
@rem
@rem The script also assumes that the current directory is %sb_dir%
@rem
@rem The script builds NifTK in release mode with testing on.

@rem ***** Attempt to enable command extensions *****
@setlocal EnableExtensions 

@rem *****  Setting localised variables - Change these to match your system!!  *****
@set "VS_LOCATION=c:\Program Files (x86)\Microsoft Visual Studio 11.0"
@set "CMAKE_LOCATION=c:\Program Files\CMake\bin"
@set "PUTTY_LOCATION=c:\Program Files (x86)\PuTTY\"
@set "OPENSSL_LOCATION=c:\OpenSSL-Win64\bin\"
@set "QT_LOCATION=c:\Qt\4.8.7\bin\"
@set "QTDIR=c:\Qt\4.8.7\bin\"
@set "GIT_LOCATION=c:\Program Files\Git\bin\"

@rem if you are cross-compiling between 64 and 32 bit then override your qt here
@rem @set "QTDIR=C:\Qt\Qt-4.8.4-x86-vc10"
@rem @set "PATH=%QTDIR%\bin;%PATH%"

@set GIT_SSL_NO_VERIFY=1

@rem *****  Set your build type 64bit/32bit  *****
@set "BTYPE=x64"
@rem @set "BTYPE=Win32"

@rem *****  Set your Visual Studio Version  *****
@set "VSVER=devenv.com"
@rem @set "VSVER=VCExpress.exe"

@rem *****  Set your CMake generator - the tool you're going to use to build NifTK  *****
@set "CMAKE_GENERATOR=Visual Studio 11 Win64"

@rem ***** Possible options are the following: ***** 
@rem NMake Makefiles             = Generates NMake makefiles.
@rem NMake Makefiles JOM         = Generates JOM makefiles.
@rem Visual Studio 11            = Generates Visual Studio 11 project files. Corresponding to Visual Studio 2012.
@rem Visual Studio 11 Win64      = Generates Visual Studio 11 Win64 project files. Corresponding to Visual Studio 2012.
@rem Visual Studio 12            = Generates Visual Studio 12 project files. Corresponding to Visual Studio 2013.
@rem Visual Studio 12 Win64      = Generates Visual Studio 12 Win64 project files. Corresponding to Visual Studio 2013.

@set "build_log=%sb_dir%/build.log"

@echo Visual Studio location: %VS_LOCATION%
@echo Visual Studio version:  %VSVER%
@echo CMake location:         %CMAKE_LOCATION%
@echo Source folder:          %src_dir%
@echo Build folder:           %sb_dir%
@echo Build log:              %build_log%
@echo CMake generator:        %CMAKE_GENERATOR%

@rem stop visual studio recycling already running instances of msbuild.exe. we want clean ones.
@rem http://stackoverflow.com/questions/12174877/visual-studio-2012-rtm-has-msbuild-exe-in-memory-after-close
@set MSBUILDDISABLENODEREUSE=1

@if ["%BTYPE%"] == ["x64"] (
  @call "%VS_LOCATION%\VC\bin\amd64\vcvars64.bat"
) else (
  @rem *****  Setting environmental variables for Win32  *****
  @call "%VS_LOCATION%\VC\bin\vcvars32.bat"
)

@set BCONF=Release

@rem *****  Configure the current VS Config Script  *****
@set "VSCONFSTRING=%BCONF%^|%BTYPE%"
@echo Current VS Config string is: %VSCONFSTRING%

@rem *****  Run CMake  *****
@set "PATH=%CMAKE_LOCATION%;%PATH%"
call "%CMAKE_LOCATION%\cmake.exe" -DCMAKE_BUILD_TYPE=%BCONF% -DEP_BASE:PATH=%ep_dir% -DEP_DIRECTORY_PER_VERSION:BOOL=ON -DEP_ALWAYS_USE_INSTALL_DIR:BOOL=OFF -DDESIRED_QT_VERSION:STRING=4 -DOPENCV_WITH_FFMPEG=ON -DNIFTK_Apps/NiftyView:BOOL=ON -DNIFTK_Apps/NiftyIGI:BOOL=ON -DNIFTK_USE_CUDA=OFF -DNIFTK_USE_GIT_PROTOCOL=ON -DBUILD_TESTING=ON -DBUILD_COMMAND_LINE_PROGRAMS=ON -DBUILD_COMMAND_LINE_SCRIPTS=ON -DNIFTK_GENERATE_DOXYGEN_HELP=ON -DNIFTYLINK_CHECK_COVERAGE=ON -G "%CMAKE_GENERATOR%" "%src_dir%"
@if %ERRORLEVEL% NEQ 0 exit /B 1

@rem *****  Run Visual Studio to build the current build conf  *****
"%VS_LOCATION%\Common7\IDE\%VSVER%" /build %BCONF% /project ALL_BUILD /projectconfig %VSCONFSTRING% %sb_dir%\NIFTK-SUPERBUILD.sln | "%GIT_LOCATION%\..\usr\bin\tee.exe" %build_log% 2>&1
@if %ERRORLEVEL% NEQ 0 exit /B 2

@rem *****  Searching for "Build Failed" string in the VS log  *****
@setlocal EnableDelayedExpansion
@set "search=^.*Build.FAILED.*$"
@findstr /r /c:"!search!" "%build_log%" >nul
@if %ERRORLEVEL% EQU 0 (
  @echo "Build error found."
  exit /B 2
)

@endlocal
@endlocal
