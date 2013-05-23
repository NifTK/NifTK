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

find_program(CTEST_GIT_COMMAND NAMES git)
find_program(CTEST_MAKE_COMMAND NAMES make)

set(CTEST_BRANCH_NAME "dev")
set(CTEST_MAKE_OPTIONS "-j 4")

set(CTEST_SOURCE_DIRECTORY "$ENV{BASE}/NifTK")
set(CTEST_BINARY_DIRECTORY "$ENV{BASE}/NifTK-SuperBuild-Release/NifTK-build")
set(NIFTK_CACHE_FILE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
set(INITIAL_CMAKECACHE_OPTIONS "${INITIAL_CMAKECACHE_OPTIONS} NIFTK_CACHE_FILE:INTERNAL=${NIFTK_CACHE_FILE}")

set(CTEST_COMMAND "ctest -D Continuous")
set(CTEST_CMAKE_COMMAND "cmake")
set(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND} pull origin ${CTEST_BRANCH_NAME}")
set(CTEST_BUILD_COMMAND "${CTEST_MAKE_COMMAND} ${CTEST_MAKE_OPTIONS}")

# build for 720 minutes eg. 06:00 am to 18:00 pm, 12 hours.
set (CTEST_CONTINUOUS_DURATION 720)

# check every 5 minutes
set (CTEST_CONTINUOUS_MINIMUM_INTERVAL 5)

# Wipe binaries each time
set (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)
