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

SET (CTEST_BRANCH_NAME "dev")
SET (CTEST_SOURCE_DIRECTORY "$ENV{BASE}/NifTK")
SET (CTEST_BINARY_DIRECTORY "$ENV{BASE}/NifTK-SuperBuild-Release/NifTK-build")
SET (CTEST_COMMAND "ctest -D Continuous")
SET (CTEST_CMAKE_COMMAND "cmake")
FIND_PROGRAM(CTEST_GIT_COMMAND NAMES git)
SET(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND} pull origin ${CTEST_BRANCH_NAME}")
SET(NIFTK_CACHE_FILE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
SET(INITIAL_CMAKECACHE_OPTIONS "${INITIAL_CMAKECACHE_OPTIONS} NIFTK_CACHE_FILE:INTERNAL=${NIFTK_CACHE_FILE}")

# build for 900 minutes eg. 06:00 am to 21:00 pm, 15 hours.
SET (CTEST_CONTINUOUS_DURATION 900)

# check every 5 minutes
SET (CTEST_CONTINUOUS_MINIMUM_INTERVAL 15)

# Wipe binaries each time
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)
