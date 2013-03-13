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

##################################################################################
# The assumption here is that the overnight build runs 21:00 - 05:59 (at
# the latest). We then run the Continuous build throughout the day
# on a pre-existing SuperBuild, so we can avoid re-building the whole lot.
# As a result, we can assume we are running off a fully configured CMakeCache.txt.
#
# A second assumption, is that the variable BASE is defined as the directory
# containing the SuperBuild, and should therefore be specified in any cron
# that runs this script. 
##################################################################################

FIND_PROGRAM(CTEST_GIT_COMMAND NAMES git)
FIND_PROGRAM(CTEST_MAKE_COMMAND NAMES make)

SET(CTEST_BRANCH_NAME "dev")
SET(CTEST_MAKE_OPTIONS "-j 4")

<<<<<<< HEAD
# If you switch from Debug to Release, the tests arent built!
SET (CTEST_INITIAL_CACHE "
BUILDNAME:STRING=Linux-Centos-g++-4.1.2-20080704-static
SITE:STRING=cmicdev
BUILD_SHARED_LIBS:STRING=OFF
BUILD_GUI:STRING=ON
CMAKE_BUILD_TYPE:STRING=Debug
BUILD_NIFTK_PROTOTYPE:STRING=OFF
USE_QT:STRING=ON
USE_VTK:STRING=ON
NIFTK_BOOST_GENERIC_PATH:FILEPATH=$ENV{HOME}/niftk/static/boost
NIFTK_ITK_GENERIC_PATH:FILEPATH=$ENV{HOME}/niftk/static/itk
NIFTK_FFTWINSTALL:FILEPATH=$ENV{HOME}/niftk/static/fftw
NIFTK_LOG4CPLUSINSTALL:FILEPATH=$ENV{HOME}/niftk/static/log4cplus
NIFTK_BOOSTINSTALL:FILEPATH=$ENV{HOME}/niftk/static/boost
NIFTK_INSTALL_PREFIX:FILEPATH=$ENV{HOME}/niftk/static
NIFTK_LINK_PREFIX:FILEPATH=$ENV{HOME}/niftk/static
NIFTK_BASE_NAME:STRING=niftk
Log4cplus_INCLUDE_DIR:PATH=$ENV{HOME}/niftk/static/log4cplus/include
Log4cplus_LIBRARIES:FILEPATH=$ENV{HOME}/niftk/static/log4cplus/lib/liblog4cplus.a
ITK_DIR:PATH=$ENV{HOME}/niftk/static/itk/lib/InsightToolkit
VTK_DIR:PATH=$ENV{HOME}/niftk/static/vtk/lib/vtk-5.6
FFTW_INCLUDE_DIR:PATH=$ENV{HOME}/niftk/static/fftw/include
FFTW_LIBRARIES:FILEPATH=$ENV{HOME}/niftk/static/fftw/lib/libfftw3.a
Boost_DATE_TIME_LIBRARY:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_date_time.a
Boost_DATE_TIME_LIBRARY_RELEASE:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_date_time.a
Boost_FILESYSTEM_LIBRARY:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_filesystem.a
Boost_FILESYSTEM_LIBRARY_RELEASE:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_filesystem.a
Boost_INCLUDE_DIR:PATH=$ENV{HOME}/niftk/static/boost/include
Boost_LIBRARY_DIRS:PATH=$ENV{HOME}/niftk/static/boost/lib
Boost_PROGRAM_OPTIONS_LIBRARY:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_program_options.a
Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_program_options.a
Boost_SYSTEM_LIBRARY:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_system.a
Boost_SYSTEM_LIBRARY_RELEASE:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_system.a
Boost_RANDOM_LIBRARY:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_random.a
Boost_RANDOM_LIBRARY_RELEASE:FILEPATH=$ENV{HOME}/niftk/static/boost/lib/libboost_random.a
")
=======
SET(CTEST_SOURCE_DIRECTORY "$ENV{BASE}/NifTK")
SET(CTEST_BINARY_DIRECTORY "$ENV{BASE}/NifTK-SuperBuild-Release/NifTK-build")
SET(NIFTK_CACHE_FILE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
SET(INITIAL_CMAKECACHE_OPTIONS "${INITIAL_CMAKECACHE_OPTIONS} NIFTK_CACHE_FILE:INTERNAL=${NIFTK_CACHE_FILE}")
>>>>>>> 5c51d8dfbe3ff80cae1a31bb9fa3094663e17be4

SET(CTEST_COMMAND "ctest -D Continuous")
SET(CTEST_CMAKE_COMMAND "cmake")
SET(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND} pull origin ${CTEST_BRANCH_NAME}")
SET(CTEST_BUILD_COMMAND "${CTEST_MAKE_COMMAND} ${CTEST_MAKE_OPTIONS}")

# build for 720 minutes eg. 06:00 am to 18:00 pm, 12 hours.
SET (CTEST_CONTINUOUS_DURATION 720)

# check every 5 minutes
SET (CTEST_CONTINUOUS_MINIMUM_INTERVAL 5)

# Wipe binaries each time
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)
