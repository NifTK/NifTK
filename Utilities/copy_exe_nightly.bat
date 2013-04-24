@set "USERNAME=***username***"
@set "PASSWORD=***password***"
@set "BUILDTAG=win7-64-vs10"

@REM ***** Some delay... *****
@ping 1.1.1.1 -n 1 -w 60000

@REM ***** Find the installer executable and pscp it to Matt's machine: *****
@cd %BUILDPATH%\NIFTK-build

@setlocal enableextensions 

@dir /b | find ".exe">%BUILD_LOCATION%\_t.txt
@SET /P _fileName=<%BUILD_LOCATION%\_t.txt

@echo ***** The current filename is: %_fileName% *****
@del %BUILD_LOCATION%\_t.txt
@"%PUTTY_LOCATION%\pscp.exe" -v -batch -pw %PASSWORD% -scp %_fileName% %USERNAME%@logno.cs.ucl.ac.uk:/tmp/niftk/niftk-nightly-%BUILDTAG%.exe >"%BUILD_LOCATION%\log_pscp_nightly.txt"

@endlocal 