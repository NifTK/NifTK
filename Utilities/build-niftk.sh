#!/bin/bash

#==============================================================================
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
#==============================================================================

# -----------------------------------------------------------------------------
# Setting defaults
# -----------------------------------------------------------------------------

build_root="."
source_dir="NifTK"
build_type="Release"
branch="dev"
commit_time="now"
threads=1
do_coverage=false
do_memcheck=false
use_gcc44=false
build_testing=true
build_docs=true
build_niftysim=false
build_command_line_tools=true
build_all_apps=true
build_midas=true
build_igi=true
build_niftyview=true
ctest_type="Nightly"
make_install=false
install_prefix="/usr/local"
make_package=false

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

function run_command()
{
  if [ $# -eq 1 ]
  then
    log_file=/dev/stdout
  elif [ $# -eq 2 ]
  then
    log_file="${log_path}/$2"
  else
    echo "ERROR: Missing command or two many options in the build script."
    return
  fi
  echo "Running '$1'" >> ${log_file}
  # This appends both stdout and stderr to the log file.
  eval $1 >> ${log_file} 2>&1
  if [ $? -ne 0 ]
  then
    echo "ERROR: command \"$1\" returned with error code $?" >> ${log_file}
  fi
}

print_usage() {
  echo "
Simple bash script to run a full automated build.
Assumes git, qt, cmake, authentication credentials, valgrind, in fact everything are already present and valid in the current shell.

Usage:

    build-niftk.sh [options] [source directory] [build directory] [log directory]


Default source directory is 'NifTK'.
Default build directory is '<source directory>-${build type}'
Default log directory is '<build-directory>-logs'.

Options:

    --build-root <directory>        Sets the build root directory, in which the source, build and log
                                    directories will be created. Defaults to '.'.

    --build-type <build type>       Sets the build type go the given value. Valid values are 'Release' and 'Debug'.
                                    Default is 'Release'.

    -r, --release                   Sets the build type to Release (default).

    -d, --debug                     Sets the build type to Debug.

    -b, --branch <branch>           Checks out the given branch. Default is 'dev'.

    -t, --time <time>               Checks out the commit from the given time. Default is 'now'.

    -j, --threads <number>          Does a parallel build with the given number of threads. Default is 1.

    -c, --coverage                  Does coverage testing.

    -v, --valgrind                  Does memory checks with valgrind.

    --gcc44                         Uses gcc4.4. Deprecated. Set the CC and CXX variables, instead.

    --no-testing                    Does not build the tests.

    --no-docs                       Does not build the documentation pages.

    --build-niftysim                Builds NiftySim. (Switched off by default.)

    --no-command-line-tools         Does not build the command line applications and scripts.

    --no-all-apps                   Does not build all the applications.

    --no-midas                      Does not build NiftyMIDAS.

    --no-igi                        Does not build NiftyIGI.

    --no-niftyview                  Does not build NiftyView.

    --ctest-type                    CTest type. Valid values are 'Nightly', 'Continuous' and 'Experimental'.
                                    Default is 'Nightly'.

    -i, --install                   Installs NifTK to the directory specified by the installation prefix.

    --prefix <directory>            Installation prefix. Default is '/usr/local'.

    -p, --package                   Creates installer packages.

"
  exit
}

print_options() {
  echo "
build-niftk.sh has been called with the following command:

  $command_line

Options:

Directories:

  build root:            $build_root
  source directory:      $source_dir
  build directory:       $build_dir
  log directory:         $log_dir

Build options:

  branch:                $branch
  commit time:           $commit_time
  build type:            $build_type
  threads:               $threads
  gcc44:                 $use_gcc44

Components:

  testing:               $build_testing
  documentation:         $build_docs
  NiftySim:              $build_niftysim
  command line tools:    $build_command_line_tools
  all apps:              $build_all_apps
  NiftyMIDAS:            $build_midas
  NiftyIGI:              $build_igi
  NiftyView:             $build_niftyview

Test options:

  coverage:              $do_coverage
  valgrind:              $do_memcheck
  ctest type:            $ctest_type

Install options:

  install prefix:        $install_prefix
  make install:          $make_install
  make package:          $make_package

"
}

check_next_arg() {
  if [ $# -lt 2 ]
  then
    echo "Missing argument."
    print_usage
  fi
  next_arg="$2"
  if [ ${#next_arg} -eq 0 ]
  then
    echo "Value is missing for option $1."
    print_usage
  fi
  if [ "${next_arg:0:1}" == "-" ]
  then
    echo "Value is missing for option $1."
    print_usage
  fi
}

# -----------------------------------------------------------------------------
# Processing arguments
# -----------------------------------------------------------------------------

command_line="$0 $@"

# Until there is any option.
while [ $# -gt 0 ] && [ ${1:0:1} == "-" ]
do
  if [ "$1" == "--build-root" ]
  then
    check_next_arg ${@}
    build_root="$2"
    if [ ! -d "${build_root}" ]
    then
      echo "ERROR: Build root directory does not exist."
      exit 1
    fi
    shift 2
  elif [ "$1" == "--build-type" ]
  then
    check_next_arg ${@}
    build_type="$2"
    shift 2
  elif [ "$1" == "-r" ] || [ "$1" == "--release" ]
  then
    build_type="Release"
    shift 1
  elif [ "$1" == "-d" ] || [ "$1" == "--debug" ]
  then
    build_type="Debug"
    shift 1
  elif [ "$1" == "-b" ] || [ "$1" == "--branch" ]
  then
    check_next_arg ${@}
    branch="$2"
    shift 2
  elif [ "$1" == "-t" ] || [ "$1" == "--time" ]
  then
    check_next_arg ${@}
    commit_time="$2"
    shift 2
  elif [ "$1" == "-j" ] || [ "$1" == "--threads" ]
  then
    check_next_arg ${@}
    threads="$2"
    shift 2
  elif [ "$1" == "-c" ] || [ "$1" == "--coverage" ]
  then
    do_coverage=true
    shift 1
  elif [ "$1" == "-v" ] || [ "$1" == "--valgrind" ]
  then
    do_memcheck=true
    shift 1
  elif [ "$1" == "--gcc44" ]
  then
    use_gcc44=true
    shift 1
  elif [ "$1" == "--no-testing" ]
  then
    build_testing=false
    shift 1
  elif [ "$1" == "--no-docs" ]
  then
    build_docs=false
    shift 1
  elif [ "$1" == "--build-niftysim" ]
  then
    build_niftysim=true
    shift 1
  elif [ "$1" == "--no-command-line-tools" ]
  then
    build_command_line_tools=false
    shift 1
  elif [ "$1" == "--no-all-apps" ]
  then
    build_all_apps=false
    shift 1
  elif [ "$1" == "--no-midas" ]
  then
    build_midas=false
    shift 1
  elif [ "$1" == "--no-igi" ]
  then
    build_igi=false
    shift 1
  elif [ "$1" == "--no-niftyview" ]
  then
    build_niftyview=false
    shift 1
  elif [ "$1" == "--ctest-type" ]
  then
    check_next_arg ${@}
    ctest_type="$2"
    if [ "$ctest_type" != "Nightly" ] && [ "$ctest_type" != "Continuous" ] && [ "$ctest_type" != "Experimental" ]
    then
      print_usage
    fi
    shift 2
  elif [ "$1" == "-i" ] || [ "$1" == "--install" ]
  then
    make_install=true
    shift 1
  elif [ "$1" == "--prefix" ]
  then
    check_next_arg ${@}
    install_prefix="$2"
    shift 2
  elif [ "$1" == "-p" ] || [ "$1" == "--package" ]
  then
    make_package=true
    shift 1
  elif [ "$1" == "-h" ] || [ "$1" == "--help" ]
  then
    print_usage
  else
    echo "Unknown argument: $1"
    print_usage
  fi
done

if [ $# -gt 0 ]
then
  source_dir="$1"
  shift 1
fi

if [ $# -gt 0 ]
then
  build_dir="$1"
  shift 1
else
  build_dir="${source_dir}-${build_type}"
fi

if [ $# -gt 0 ]
then
  log_dir="$1"
  shift 1
else
  log_dir="${build_dir}-logs"
fi

source_path=${build_root}/${source_dir}
build_path=${build_root}/${build_dir}
log_path=${build_root}/${log_dir}

if [ $# -gt 0 ]
then
  print_usage
fi

# -----------------------------------------------------------------------------
# Set up the environment
# -----------------------------------------------------------------------------

NIFTK_SUPERBUILD_DIR="${build_root}/${build_dir}"
NIFTK_DIR=$NIFTK_SUPERBUILD_DIR/NifTK-build
ITK_DIR=$NIFTK_SUPERBUILD_DIR/ITK-build
VTK_DIR=$NIFTK_SUPERBUILD_DIR/VTK-build
GDCM_DIR=$NIFTK_SUPERBUILD_DIR/GDCM-build
CTK_DIR=$NIFTK_SUPERBUILD_DIR/CTK-build/CTK-build
MITK_DIR=$NIFTK_SUPERBUILD_DIR/MITK-build/MITK-build

export NIFTK_DRC_ANALYZE=ON
export GIT_SSL_NO_VERIFY=1

export LD_LIBRARY_PATH="$BOOST_DIR:$ITK_DIR/bin:$VTK_DIR/bin:$GDCM_DIR/bin:$CTK_DIR/bin:$MITK_DIR/bin:$MITK_DIR/bin/plugins:$NIFTK_DIR/bin:$NIFTK_DIR/bin/plugins"
export DYLD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# -----------------------------------------------------------------------------
# Clean old directories
# -----------------------------------------------------------------------------

if [ -d ${log_path} ]
then
  # Deleting old log directory.
  rm -rf ${log_path}
fi

mkdir -p ${log_path}

if [ -d "${source_path}" ] && [ "${ctest_type}" == "Nightly" ]
then
  run_command "echo Deleting source code directory" 0-clean.log
  run_command "\rm -rf ${source_path}" 0-clean.log
fi

if [ -d "${build_path}" ] && [ "${ctest_type}" == "Nightly" ]
then
  run_command "echo Deleting old build directory ${build_path}" 0-clean.log
  run_command "\rm -rf ${build_path}" 0-clean.log
fi

run_command "mkdir -p ${build_path}" 0-clean.log

##DATE=`date -u +%F`

# -----------------------------------------------------------------------------
# Composing cmake and ctest command line
# -----------------------------------------------------------------------------

cmake_args="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_INSTALL_PREFIX=${install_prefix}"

if $do_coverage
then
  cmake_args="${cmake_args} -DNIFTK_CHECK_COVERAGE=ON"
else
  cmake_args="${cmake_args} -DNIFTK_CHECK_COVERAGE=OFF"
fi

if $do_memcheck
then
  ctest_command="make clean ; ctest -D ${ctest_type}Start ; ctest -D ${ctest_type}Update ; ctest -D ${ctest_type}Configure ; ctest -D ${ctest_type}Build ; ctest -D ${ctest_type}Test ; ctest -D ${ctest_type}Coverage ; ctest -D ${ctest_type}MemCheck ; ctest -D ${ctest_type}Submit"
else
  ctest_command="make clean ; ctest -D ${ctest_type}"
fi

if $use_gcc44
then
  cmake_args="${cmake_args} -DCMAKE_C_COMPILER=/usr/bin/gcc44 -DCMAKE_CXX_COMPILER=/usr/bin/g++44"
fi

if $build_docs
then
  cmake_args="${cmake_args} -DNIFTK_GENERATE_DOXYGEN_HELP=ON"
else
  cmake_args="${cmake_args} -DNIFTK_GENERATE_DOXYGEN_HELP=OFF"
fi

if $build_testing
then
  cmake_args="${cmake_args} -DBUILD_TESTING=ON"
else
  cmake_args="${cmake_args} -DBUILD_TESTING=OFF"
fi

if $build_niftysim
then
  cmake_args="${cmake_args} -DBUILD_NIFTYSIM=ON"
else
  cmake_args="${cmake_args} -DBUILD_NIFTYSIM=OFF"
fi

if $build_command_line_tools
then
  cmake_args="${cmake_args} -DBUILD_COMMAND_LINE_PROGRAMS=ON -DBUILD_COMMAND_LINE_SCRIPTS=ON"
else
  cmake_args="${cmake_args} -DBUILD_COMMAND_LINE_PROGRAMS=OFF -DBUILD_COMMAND_LINE_SCRIPTS=OFF"
fi

if $build_all_apps
then
  cmake_args="${cmake_args} -DNIFTK_BUILD_ALL_APPS=ON"
else
  cmake_args="${cmake_args} -DNIFTK_BUILD_ALL_APPS=OFF"
fi

if $build_midas
then
  cmake_args="${cmake_args} -DNIFTK_Apps/NiftyMIDAS=ON"
else
  cmake_args="${cmake_args} -DNIFTK_Apps/NiftyMIDAS=OFF"
fi

if $build_igi
then
  cmake_args="${cmake_args} -DNIFTK_Apps/NiftyIGI=ON"
else
  cmake_args="${cmake_args} -DNIFTK_Apps/NiftyIGI=OFF"
fi

if $build_niftyview
then
  cmake_args="${cmake_args} -DNIFTK_Apps/NiftyView=ON"
else
  cmake_args="${cmake_args} -DNIFTK_Apps/NiftyView=OFF"
fi

# -----------------------------------------------------------------------------
# Do the job
# -----------------------------------------------------------------------------

echo "Build started at `date` on `hostname -f`." > ${log_dir}/1-start.log
print_options >> ${log_path}/1-start.log

run_command "git clone https://cmicdev.cs.ucl.ac.uk/git/NifTK ${source_path}" 2-clone.log
cd ${source_path}
# For some reason the time-based checkout works only if the branch has already been checked out once.
run_command "git checkout $branch" 3-checkout.log
run_command "git checkout $branch@{$commit_time}" 3-checkout.log
cd ${build_path}
run_command "cmake ${cmake_args} ${source_path}" 4-cmake.log
run_command "make -j ${threads}" 5-build.log
cd NifTK-build
# Note that the submit task fails with http timeout, but we want to carry on regardless to get to the package bit.
run_command "${ctest_command}" 6-ctest.log

if $make_install
then
  run_command "make install" 7-install.log
fi

if $make_package
then
  run_command "make package" 8-package.log
fi

echo "Finished at `date`" > ${log_path}/9-finish.log
