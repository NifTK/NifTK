#!/bin/bash

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

# If the NIFTK_MAKE_PACKAGE variable is defined then a 'make package'
# command is executed.

# If the NIFTK_INSTALL_PREFIX variable is defined then a 'make install'
# command is executed.

# If the NIFTK_CTEST_TYPE variable is defined then it determines the type of test
# to run. Valid values are "Nightly", "Continuous" and "Experimental".

function run_command()
{
  echo "Running \"$1\""
  eval $1
  if [ $? -ne 0 ]; then
    echo "ERROR: command \"$1\" returned with error code $?"
  fi
}

if [ $# -ne 5 ] && [ $# -ne 6 ] && [ $# -ne 7 ] && [ $# -ne 8 ]
then
  echo "Simple bash script to run a full automated build. "
  echo "Assumes git, qt, cmake, authentication credentials, valgrind, in fact everything are already present and valid in the current shell."  
  echo "Does a two pass checkout. It checks out NifTK at the time you run this script, then does an update to the correct time to run unit tests, which means the dashboard shows the wrong number of updates."
  echo "Usage: NifTKUnixBuild.sh [Debug|Release] <number_of_threads> [cov|nocov to control coverage] [val|noval to control valgrind] [gcc44|nogcc44 to use gcc4] [branch] [source directory] [build directory]"
  exit -1
fi

TYPE=$1
THREADS=$2
COVERAGE=$3
MEMCHECK=$4
GCC4=$5
if [ $# -ge 6 ]
then
  BRANCH=$6
else
  BRANCH=dev
fi

if [ $# -ge 7 ]
then
  SOURCE_DIR=$7
else
  SOURCE_DIR=NifTK
fi

if [ $# -eq 8 ]
then
  BINARY_DIR=$8
else
  BINARY_DIR=${SOURCE_DIR}-SuperBuild-${TYPE}
fi

if [ "${TYPE}" != "Debug" -a "${TYPE}" != "Release" ]; then
  echo "First argument after NifTKUnixBuild.sh must be either Debug or Release."
  exit -2
fi

if [ -d "${SOURCE_DIR}" ]; then
  echo "Deleting source code folder"
  run_command "\rm -rf ${SOURCE_DIR}"
fi

if [ -d "${BINARY_DIR}" ]; then
  echo "Deleting old build in folder ${BINARY_DIR}"
  run_command "\rm -rf ${BINARY_DIR}"
fi

DATE=`date -u +%F`
if [ "${COVERAGE}" = "cov" ]; then
  COVERAGE_ARG="-DNIFTK_CHECK_COVERAGE=ON"
fi

if [ "${GCC4}" = "gcc44" ]; then
  GCC4_ARG="-DCMAKE_C_COMPILER=/usr/bin/gcc44 -DCMAKE_CXX_COMPILER=/usr/bin/g++44"
fi

if [ ! -z "${NIFTK_CTEST_TYPE}" ]
then
  CTestType=${NIFTK_CTEST_TYPE}
else
  CTestType="Nightly"
fi

if [ "${MEMCHECK}" = "val" ]; then
  BUILD_COMMAND="make clean ; ctest -D ${CTestType}Start ; ctest -D ${CTestType}Update ; ctest -D ${CTestType}Configure ; ctest -D ${CTestType}Build ; ctest -D ${CTestType}Test ; ctest -D ${CTestType}Coverage ; ctest -D ${CTestType}MemCheck ; ctest -D ${CTestType}Submit"
else
  BUILD_COMMAND="make clean ; ctest -D ${CTestType}"
fi

if [ ! -z "${NIFTK_INSTALL_PREFIX}" ]
then
  NIFTK_INSTALL_OPTIONS="-DCMAKE_INSTALL_PREFIX=${NIFTK_INSTALL_PREFIX}"
fi

run_command "git clone https://cmicdev.cs.ucl.ac.uk/git/NifTK ${SOURCE_DIR}"
run_command "cd ${SOURCE_DIR}"
run_command "git checkout ${BRANCH}"
run_command "cd .."
run_command "mkdir ${BINARY_DIR}"
run_command "cd ${BINARY_DIR}"
run_command "cmake ../${SOURCE_DIR} ${COVERAGE_ARG} ${GCC4_ARG} -DCMAKE_BUILD_TYPE=${TYPE} ${NIFTK_INSTALL_OPTIONS} -DNIFTK_BUILD_ALL_APPS=ON -DBUILD_TESTING=ON -DBUILD_COMMAND_LINE_PROGRAMS=ON -DBUILD_COMMAND_LINE_SCRIPTS=ON -DNIFTK_GENERATE_DOXYGEN_HELP=ON"
run_command "make -j ${THREADS}"
run_command "cd NifTK-build"
run_command "${BUILD_COMMAND}" # Note that the submit task fails with http timeout, but we want to carry on regardless to get to the package bit.

if [ "${TYPE}" = "Release" -a ! -z "${NIFTK_MAKE_PACKAGE}" ]; then
  run_command "make package"
fi

if [ ! -z "${NIFTK_INSTALL_PREFIX}" ]
then
  run_command "make install"
fi
