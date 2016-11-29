@rem The script is invoked from the gitlab-ci.yml job without arguments.
@rem The script assumes that the following environment variables are set:
@rem
@rem   - src_dir: points to the NifTK source folder
@rem   - ep_dir: points to the folder where the external projects are (to be) built
@rem   - sb_dir: points to the folder where NifTK is to be built (superbuild folder)
@rem
@rem The script also assumes that the current directory is %sb_dir%
@rem
@rem The script runs the test for a NifTK release mode build.

@rem ***** Attempt to enable command extensions *****
@setlocal EnableExtensions 

@rem *****  Setting localised variables - Change these to match your system!!  *****
@set "VS_LOCATION=c:\Program Files (x86)\Microsoft Visual Studio 11.0"
@set "CMAKE_LOCATION=c:\Program Files\CMake\bin"
@set "OPENSSL_LOCATION=c:\OpenSSL-Win64\bin\"

@rem if you are cross-compiling between 64 and 32 bit then override your qt here
@rem @set "QTDIR=C:\Qt\Qt-4.8.4-x86-vc10"
@rem @set "PATH=%QTDIR%\bin;%PATH%"

@rem *****  Set your build type 64bit/32bit  *****
@set "BTYPE=x64"
@rem @set "BTYPE=Win32"

@echo CMake location:         %CMAKE_LOCATION%
@echo Source folder:          %src_dir%
@echo Build folder:           %sb_dir%

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

@rem  *****  Set PATH and Environment for NifTK  *****
@cd /d "%pb_dir%"
@set CL=/D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_DEPRECATE
@set LINK=/LARGEADDRESSAWARE
@set NIFTK_DRC_ANALYZE=ON

@setlocal EnableDelayedExpansion

@set "BATFILEPATH=%pb_dir%\bin\startNiftyView_%BCONF%.bat"
@set PATHSTRING1=" "
@set PATHSTRING2=" "

@echo Reading %BATFILEPATH%....

@rem ***** Extracting path via equal sign (???) from startNiftyView_%BCONF%.bat *****
@rem ***** As of 30/11/2015 the path is divided into two parts; we need to take both for proper linking.

@set /a counter=1
@for /f ^"usebackq^ eol^=^

^ delims^=^" %%a in (%BATFILEPATH%) do @(
        @if "!counter!"==4 goto :eof
        @set var!counter!=%%a
        @set /a counter+=1
)

@rem ***** Taking the first two lines and removing PATH=
@set PATHSTRING1=%var1:/=\%
@set PATHSTRING1=%PATHSTRING1:PATH=%
@set PATHSTRING1=%PATHSTRING1:~1,-2%

@set PATHSTRING2=%var2:/=\%
@set PATHSTRING2=%PATHSTRING2:PATH=%
@set PATHSTRING2=%PATHSTRING2:~1,-2%

@PATH=%PATHSTRING2%%PATHSTRING1%%SystemRoot%;%SystemRoot%\system32;%SystemRoot%\System32\Wbem;%OPENSSL_LOCATION%

@if defined CUDA_PATH PATH=%PATH%;%CUDA_PATH%\bin

@echo The current system path:
@echo %PATH%

"%CMAKE_LOCATION%\ctest.exe" -C %BCONF% -E CTE-Stream -S CTestContinuous.cmake -V
@if %ERRORLEVEL% NEQ 0 exit /B 4

@endlocal
@endlocal
