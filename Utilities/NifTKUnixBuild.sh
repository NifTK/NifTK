#!/bin/bash

#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-06-01 09:38:00 +0100 (Wed, 01 Jun 2011) $ 
#  Revision          : $Revision: 6322 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# If the NIFTK_INSTALL_PREFIX variable is defined then a 'make install'
# command is executed as well.

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

if [ "${MEMCHECK}" = "val" ]; then
  BUILD_COMMAND="make clean ; ctest -D NightlyStart ; ctest -D NightlyUpdate ; ctest -D NightlyConfigure ; ctest -D NightlyBuild ; ctest -D NightlyTest ; ctest -D NightlyCoverage ; ctest -D NightlyMemCheck ; ctest -D NightlySubmit"
else
  BUILD_COMMAND="make clean ; ctest -D Nightly"
fi  

if [ ! -z ${NIFTK_INSTALL_PREFIX} ]
then
  NIFTK_INSTALL_OPTIONS="-DCMAKE_INSTALL_PREFIX=${NIFTK_INSTALL_PREFIX}"
fi

run_command "git clone https://cmicdev.cs.ucl.ac.uk/git/NifTK ${SOURCE_DIR}"
run_command "cd ${SOURCE_DIR}"
run_command "git checkout -b ${BRANCH} origin/${BRANCH}"
run_command "cd .."
run_command "mkdir ${BINARY_DIR}"
run_command "cd ${BINARY_DIR}"
run_command "cmake ../${SOURCE_DIR} ${COVERAGE_ARG} ${GCC4_ARG} -DCMAKE_BUILD_TYPE=${TYPE} ${NIFTK_INSTALL_OPTIONS} -DNIFTK_BUILD_ALL_APPS=ON -DBUILD_TESTING=ON -DBUILD_COMMAND_LINE_PROGRAMS=ON -DBUILD_COMMAND_LINE_SCRIPTS=ON -DNIFTK_GENERATE_DOXYGEN_HELP=ON"
run_command "make -j ${THREADS}"
run_command "cd NifTK-build"
run_command "${BUILD_COMMAND}" # Note that the submit task fails with http timeout, but we want to carry on regardless to get to the package bit.

if [ "${TYPE}" = "Release" ]; then
  run_command "make package"
fi

if [ ! -z ${NIFTK_INSTALL_PREFIX} ]
then
  run_command "make install"
fi
