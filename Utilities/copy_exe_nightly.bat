ping 1.1.1.1 -n 1 -w 60000
pscp -v -batch -pw PASSWORD -scp d:\NifTK\NIFTK_build_release\NifTK-build\niftk-1.0.0rc1.exe USER@logno.cs.ucl.ac.uk:/tmp/niftk/niftk-nightly-win7-64.exe >log_pscp_nightly.txt
