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

function run_command()
{
  echo "Running $1, with cautious flag=$2"
  eval $1
  if [ $? -ne 0 ]; then
    if [ "$2" = "OFF" ]; then
      echo "Emergency exit due to exit code:$?"
      exit $?
    fi
  fi
}

if [ $# -ne 5 ]; then
  echo "Simple bash script to run a full automated build. "
  echo "Does a two pass checkout. It checks out NifTK at the time you run it, then does a svn update with proper revision ID etc. to run unit tests"
  echo "Assumes svn, git, qt, cmake, svn credentials, valgrind, in fact everything are already present and valid in the current shell."
  echo "Usage: UnixBuild.sh [Debug|Release] <number_of_threads> [ON|OFF to control coverage] [ON|OFF to control valgrind] [ON|OFF to build OpenCV]"
  exit -1
fi

TYPE=$1
THREADS=$2
COVERAGE=$3
MEMCHECK=$4
OPENCV=$5

if [ "${TYPE}" != "Debug" -a "${TYPE}" != "Release" ]; then
  echo "First argument after UnixBuild.sh must be either Debug or Release."
  exit -2
fi

if [ -d "NifTK" ]; then
  echo "Deleting source code folder"
  run_command "\rm -rf NifTK"
fi

FOLDER=NifTK-SuperBuild-${TYPE}
if [ -d "${FOLDER}" ]; then
  echo "Deleting old build in folder ${FOLDER}"
  run_command "\rm -rf ${FOLDER}"
fi

DATE=`date -u +%F`
if [ "${COVERAGE}" = "ON" ]; then
  COVERAGE_ARG="-DNIFTK_CHECK_COVERAGE=ON"
fi

if [ "${OPENCV} = "ON" ]; then
  OPENCV_ARG="-DBUILD_OPENCV=ON"
fi

if [ "${MEMCHECK}" = "ON" ]; then
  BUILD_COMMAND="make clean ; ctest -D NightlyStart ; ctest -D NightlyUpdate ; ctest -D NightlyConfigure ; ctest -D NightlyBuild ; ctest -D NightlyTest ; ctest -D NightlyCoverage ; ctest -D NightlyMemCheck ; ctest -D NightlySubmit"
else
  BUILD_COMMAND="make clean ; ctest -D Nightly"
fi  

run_command "svn co https://cmicdev.cs.ucl.ac.uk/svn/cmic/trunk/NifTK --non-interactive"
run_command "mkdir ${FOLDER}"
run_command "cd ${FOLDER}"
run_command "cmake ../NifTK -DCMAKE_BUILD_TYPE=${TYPE} -DBUILD_GUI=ON -DBUILD_TESTING=ON -DBUILD_COMMAND_LINE_PROGRAMS=ON -DBUILD_COMMAND_LINE_SCRIPTS=ON -DBUILD_NIFTYLINK=ON -DBUILD_OPENIGTLINK=ON -DNIFTK_GENERATE_DOXYGEN_HELP=ON ${COVERAGE_ARG} ${OPENCV_ARG}"
run_command "make -j ${THREADS}"
run_command "cd NifTK-build"
run_command "${BUILD_COMMAND} OFF"  # The submit task fails due to http timeout, so we want to carry on regardles.

if [ "${TYPE}" = "Release" ]; then
  run_command "make package"
fi






