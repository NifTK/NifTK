Notes:
------

CMakeExternals   - CMake files to drive the Superbuild process which builds dependencies as 'External' projects.
cuda             - CMake files for cuda.
Continuous*      - CMake files, to be used as templates when setting up an overnight build - OBSELETE.
Find*            - CMake files for finding packages. If you see @ variables, these get substituted for known paths.
mitkCompiler*    - CMake compiler settings borrowed from MITK project.
StartVS*         - Batch file that gets generated using the build into the build folder to assist with launching Visual Studio.
SuperBuild*      - CMake file to drive the Superbuild process
SetupCPack.cmake - CMake file containing common variables for CPack for all generators.
 