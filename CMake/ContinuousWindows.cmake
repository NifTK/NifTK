#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/


SET (CTEST_SOURCE_DIRECTORY "C:/build/Nightly/NifTK")
SET (CTEST_BINARY_DIRECTORY "C:/build/Nightly/NifTK-build")
SET (CTEST_COMMAND "ctest -D Nightly")
SET (CTEST_CMAKE_COMMAND "cmake")
SET (CTEST_SVN_COMMAND "C:/Program Files/SlikSvn/bin/svn.exe")

SET (CTEST_INITIAL_CACHE "
BUILDNAME:STRING=Win32-vs9-Nightly
SITE:STRING=Aninds-Desktop
BUILD_SUPERBUILD:BOOL=OFF
BUILD_GUI:BOOL=ON
BUILD_SHARED_LIBS:BOOL=OFF
#CMAKE_CONFIGURATION_TYPES:STRING=Debug
USE_QT:STRING=ON
USE_VTK:STRING=ON
CMAKE_BUILD_TYPE:STRING=Debug
CMAKE_CXX_FLAGS:STRING= /DWIN32 /D_WINDOWS /W3 /Zm1000 /EHsc /GR /bigobj
NIFTK_INSTALL_PREFIX:FILEPATH=C:/build/Nightly/install
NIFTK_LINK_PREFIX:FILEPATH=C:/install
NIFTK_BASE_NAME:STRING=niftk
ITK_DIR:PATH=C:/install/InsightToolkit-3.20.0/lib/InsightToolkit
VTK_DIR:PATH=C:/install/vtk-5.6.1/lib/vtk-5.6
FFTW_INCLUDE_DIR:PATH=C:/install/fftw-3.2.2.pl1-dll32
FFTW_LIBRARIES:FILEPATH=C:/install/fftw-3.2.2.pl1-dll32/libfftw3f-3.lib
Boost_ADDITIONAL_VERSIONS:STRING=1.4.7
BOOST_LIBRARYDIR:PATH=C:/install/boost/boost_1_47/lib
Boost_DATE_TIME_LIBRARY:PATH=optimized;C:/install/boost/boost_1_47/lib/libboost_date_time-vc90-mt-1_47.lib;debug;C:/install/boost/boost_1_47/lib/libboost_date_time-vc90-mt-gd-1_47.lib
Boost_DATE_TIME_LIBRARY_DEBUG:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_date_time-vc90-mt-gd-1_47.lib
Boost_DATE_TIME_LIBRARY_RELEASE:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_date_time-vc90-mt-1_47.lib
Boost_FILESYSTEM_LIBRARY:PATH=optimized;C:/install/boost/boost_1_47/lib/libboost_filesystem-vc90-mt-1_47.lib;debug;C:/install/boost/boost_1_47/lib/libboost_filesystem-vc90-mt-gd-1_47.lib
Boost_FILESYSTEM_LIBRARY_DEBUG:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_filesystem-vc90-mt-gd-1_47.lib
Boost_FILESYSTEM_LIBRARY_RELEASE:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_filesystem-vc90-mt-1_47.lib
Boost_INCLUDE_DIR:PATH=C:/install/boost/boost_1_47
Boost_LIBRARY_DIRS:PATH=optimized;C:/install/boost/boost_1_47/lib/libboost_filesystem-vc90-mt-1_47.lib;debug;C:/install/boost/boost_1_47/lib
Boost_LIB_DIAGNOSTIC_DEFINITIONS:STRING=-DBOOST_LIB_DIAGNOSTIC
Boost_PROGRAM_OPTIONS_LIBRARY:PATH=optimized;C:/install/boost/boost_1_47/lib/libboost_program_options-vc90-mt-1_47.lib;debug;C:/install/boost/boost_1_47/lib/libboost_program_options-vc90-mt-gd-1_47.lib
Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_program_options-vc90-mt-gd-1_47.lib
Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_program_options-vc90-mt-1_47.lib
Boost_SYSTEM_LIBRARY:PATH=optimized;C:/install/boost/boost_1_47/lib/libboost_system-vc90-mt-1_47.lib;debug;C:/install/boost/boost_1_47/lib/libboost_system-vc90-mt-gd-1_47.lib
Boost_SYSTEM_LIBRARY_DEBUG:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_system-vc90-mt-gd-1_47.lib
Boost_SYSTEM_LIBRARY_RELEASE:FILEPATH=C:/install/boost/boost_1_47/lib/libboost_system-vc90-mt-1_47.lib 
Boost_USE_MULTITHREADED:BOOL=ON
Boost_USE_STATIC_LIBS:BOOL=ON
")

SET (CTEST_CONTINUOUS_DURATION 30)
SET (CTEST_CONTINUOUS_MINIMUM_INTERVAL 15)
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)
SET (CTEST_SUBMIT_RETRY_COUNT 0)
SET (CTEST_SUBMIT_RETRY_DELAY 10)

