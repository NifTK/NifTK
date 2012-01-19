REM *****  Setting environmental variables for x64  *****
CALL "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\amd64\vcvarsamd64.bat"

REM *****  Cleaning off the local UCLTK repo and fetching latest source from SVN  *****
rd /s /q d:\NIFTK\NIFTK
md d:\NIFTK\NIFTK
"c:\msysgit\msysgit\bin\svn.exe" co https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NifTK --username USER --password PASSWORD

REM *****  Cleaning off the local build folder  *****
REM rd /s /q d:\NIFTK\NIFTK_build_debug
rd /s /q d:\NIFTK\NIFTK_build_release
md d:\NIFTK\NIFTK_build_release

REM *****  Adding local mods - TEMPORARY  *****
copy d:\NIFTK\CMakeCache_release.txt d:\NIFTK\NIFTK_build_release\CMakeCache.txt

REM *****  Clear buildlog  *****
cd d:\NIFTK
del buildlog_release.txt

REM *****  Run CMAKE  *****
cd d:\NIFTK\NIFTK_build_release
c:\cmake_build\bin\Release\cmake.exe  d:\NIFTK\NIFTK

REM *****  Run devenv --> Clean + Rebuild for debug|x64  *****
"C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE\devenv.exe" /build release /project ALL_BUILD /projectconfig "Release|x64" /out "d:\NIFTK\log_build-release.txt" NIFTK-SUPERBUILD.sln

REM  *****  Set PATH and Environment for UCLTK  *****
cd "d:\NIFTK\NIFTK_build_release\NIFTK-build\"
@set CL=/D_CRT_SECURE_NO_DEPRECATE /D_CRT_NONSTDC_NO_DEPRECATE
@set LINK=/LARGEADDRESSAWARE
PATH=D:\NIFTK\NIFTK_build_release\VTK-build\bin\release;D:\NIFTK\NIFTK_build_release\ITK-build\bin\release;D:\NIFTK\NIFTK_build_release\MITK-build\MITK-build\bin\release;C:\Qt\4.7.2_x64\lib\..\bin;C:\_IGSTK_STUFF\build_MITK\CTK-build\CTK-build\bin\release;D:\NIFTK\NIFTK_build_release\MITK-build\MITK-build\bin\plugins\release;D:\NIFTK\NIFTK_build_release\GDCM-build\bin\release;D:\NIFTK\NIFTK_build_release\NIFTK-build\bin\release;D:\NIFTK\NIFTK_build_release\NIFTK-build\bin\plugins\release;D:\NIFTK\NIFTK_build_release\CMakeExternals\Install\BOOST\lib;%PATH%

REM *****  Run CTEST  *****
REM ctest -T Test -T Submit -E CTE-Stream  -VV --track Nightly
REM c:\cmake_build\bin\Release\ctest.exe -C Release -E CTE-Stream -E MIDAS-Seg-RegGrowProc -E MIDAS-Seg-PropUp -E MIDAS-Seg-PropDown -D Nightly
c:\cmake_build\bin\Release\ctest.exe -C Release -E CTE-Stream -D NightlyStart >d:\NIFTK\log_ctest.txt
c:\cmake_build\bin\Release\ctest.exe -C Release -E CTE-Stream -D NightlyConfigure >>d:\NIFTK\log_ctest.txt
c:\cmake_build\bin\Release\ctest.exe -C Release -E CTE-Stream -D NightlyBuild >>d:\NIFTK\log_ctest.txt
c:\cmake_build\bin\Release\ctest.exe -C Release -E CTE-Stream -D NightlyTest >>d:\NIFTK\log_ctest.txt
c:\cmake_build\bin\Release\ctest.exe -C Release -E CTE-Stream -D NightlySubmit >>d:\NIFTK\log_ctest.txt

"C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE\devenv.exe" /build release /project PACKAGE /projectconfig "Release|x64" /out "D:\NIFTK\log_build-package.txt" NIFTK.sln

ping 1.1.1.1 -n 1 -w 60000
pscp -v -batch -pw PASSWORD -scp d:\NifTK\NIFTK_build_release\NifTK-build\niftk-1.0.0rc1.exe USER@logno.cs.ucl.ac.uk:/tmp/niftk/niftk-nightly-win7-64.exe >D:\NIFTK\log_pscp_nightly.txt