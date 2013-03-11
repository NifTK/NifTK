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

SET (CTEST_BRANCH_NAME "dev")
SET (CTEST_SOURCE_DIRECTORY "$ENV{BASE}/NifTK")
SET (CTEST_BINARY_DIRECTORY "$ENV{BASE}/NifTK-SuperBuild-Release/NifTK-build")
SET (CTEST_COMMAND "ctest -D Continuous")
SET (CTEST_CMAKE_COMMAND "cmake")
FIND_PROGRAM(CTEST_GIT_COMMAND NAMES git)
SET(CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND} pull origin ${CTEST_BRANCH_NAME}")
SET(NIFTK_CACHE_FILE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt")
SET(INITIAL_CMAKECACHE_OPTIONS "${INITIAL_CMAKECACHE_OPTIONS} NIFTK_CACHE_FILE:INTERNAL=${NIFTK_CACHE_FILE}")

# build for 720 minutes eg. 06:00 am to 18:00 pm, 12 hours.
SET (CTEST_CONTINUOUS_DURATION 720)

# check every 5 minutes
SET (CTEST_CONTINUOUS_MINIMUM_INTERVAL 5)

# Wipe binaries each time
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)
