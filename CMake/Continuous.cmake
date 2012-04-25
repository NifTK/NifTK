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
#  Last Changed      : $LastChangedDate: 2011-12-17 14:35:07 +0000 (Sat, 17 Dec 2011) $ 
#  Revision          : $Revision: 8065 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/wc/static/NifTK")
SET (CTEST_BINARY_DIRECTORY "$ENV{HOME}/wc/static/NifTK-build")
SET (CTEST_COMMAND "$ENV{HOME}/niftk/cmake/bin/ctest -D Nightly")
SET (CTEST_CMAKE_COMMAND "$ENV{HOME}/niftk/cmake/bin/cmake")
SET (CTEST_SVN_COMMAND "/usr/bin/svn")

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
")


# build for 15 minutes
SET (CTEST_CONTINUOUS_DURATION 15)

# check every 15 minutes
SET (CTEST_CONTINUOUS_MINIMUM_INTERVAL 15)

# Wipe binaries each time
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY FALSE)
