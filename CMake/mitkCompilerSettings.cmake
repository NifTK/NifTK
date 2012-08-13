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
#  Last Changed      : $LastChangedDate: 2011-07-08 16:29:16 +0100 (Fri, 08 Jul 2011) $ 
#  Revision          : $Revision: 6703 $
#  Last modified by  : $Author: ad $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Additional stuff that gets included if we are using MITK

INCLUDE(mitkFunctionCheckCompilerFlags)
INCLUDE(mitkFunctionGetGccVersion)
INCLUDE(mitkFunctionGetVersion)

# Retrieve some software versions
mitkFunctionGetVersion(${MITK_SOURCE_DIR} MITK)

# Trac 1627 - this mitkFunctionGetVersion didn't appear to work on Marc's laptop.
# We can switch back to mitkFunctionGetVersion when the project is in git.
# In the meantime we can use the following code, borrowed from NiftyReg.
# mitkFunctionGetVersion(${CMAKE_SOURCE_DIR} NIFTK_SVN)

SET (NIFTK_SVN_REVISION_ID "Unknown")
IF(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.svn)
  FIND_PACKAGE(Subversion)
  IF(Subversion_FOUND)
      Subversion_WC_INFO(${CMAKE_SOURCE_DIR} NifTK)
      SET(NIFTK_SVN_REVISION_ID ${NifTK_WC_REVISION})
  endif(Subversion_FOUND)
ENDIF()


IF(BUILD_GUI)
  MESSAGE("Qt version=${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}.${QT_VERSION_PATCH}")
  
  mitkFunctionGetVersion(${CTK_SOURCE_DIR} CTK) # We should always build off a hashtag, so this should match that in CTK.cmake
  MESSAGE("CTK version=${CTK_REVISION_ID}")
ENDIF()

# Print out other versions.
MESSAGE("BOOST version=${NIFTK_VERSION_BOOST}")                 
MESSAGE("GDCM version=${NIFTK_VERSION_GDCM}")                   
MESSAGE("DCMTK version=${NIFTK_VERSION_DCMTK}")
MESSAGE("ITK version=${NIFTK_VERSION_ITK}") 
MESSAGE("VTK version=${NIFTK_VERSION_VTK}")                     
MESSAGE("MITK version=${MITK_REVISION_ID}")

IF(BUILD_IGI)
  mitkFunctionGetVersion(${NiftyLink_SOURCE_DIR} NIFTYLINK)
  MESSAGE("NiftyLink version=${NIFTYLINK_REVISION_ID}")
ENDIF(BUILD_IGI)

MESSAGE("NIFTK version=${NIFTK_SVN_REVISION_ID}")

# MinGW does not export all symbols automatically, so no need to set flags
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW)
  #SET(VISIBILITY_CXX_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW)

if(NOT UNIX AND NOT MINGW)
  set(MITK_WIN32_FORCE_STATIC "STATIC")
endif(NOT UNIX AND NOT MINGW)

if(MITK_USE_QT)
  set(QT_QMAKE_EXECUTABLE ${MITK_QMAKE_EXECUTABLE})
  add_definitions(-DQWT_DLL)
endif()

set(NIFTK_MITK_C_FLAGS "${NIFTK_COVERAGE_C_FLAGS} ${NIFTK_ADDITIONAL_C_FLAGS}")
set(NIFTK_MITK_CXX_FLAGS "${VISIBILITY_CXX_FLAGS} ${NIFTK_COVERAGE_CXX_FLAGS} ${NIFTK_ADDITIONAL_CXX_FLAGS}")

IF(APPLE)
  SET(NIFTK_MITK_CXX_FLAGS "${NIFTK_MITK_CXX_FLAGS} -DNIFTK_OS_IS_MAC")
ENDIF(APPLE)

if(CMAKE_COMPILER_IS_GNUCXX)

  SET(MITK_GNU_COMPILER_C_WARNINGS "-Wall -Wextra -Wpointer-arith -Winvalid-pch -Wcast-align -Wwrite-strings -D_FORTIFY_SOURCE=2")
  SET(MITK_GNU_COMPILER_CXX_WARNINGS "-Woverloaded-virtual -Wold-style-cast -Wstrict-null-sentinel -Wsign-promo ")

  IF(NIFTK_VERBOSE_COMPILER_WARNINGS)
    SET(cflags "${MITK_GNU_COMPILER_C_WARNINGS}")
    SET(cxxflags "${cflags} ${MITK_GNU_COMPILER_CXX_WARNINGS}") 
  ENDIF(NIFTK_VERBOSE_COMPILER_WARNINGS)
  
  mitkFunctionCheckCompilerFlags("-fdiagnostics-show-option" cflags)
  mitkFunctionCheckCompilerFlags("-Wl,--no-undefined" cflags)
  mitkFunctionGetGccVersion(${CMAKE_CXX_COMPILER} GCC_VERSION)
  
  # With older version of gcc supporting the flag -fstack-protector-all, an extra dependency to libssp.so
  # is introduced. If gcc is smaller than 4.4.0 and the build type is Release let's not include the flag.
  # Doing so should allow to build package made for distribution using older linux distro.
  if(${GCC_VERSION} VERSION_GREATER "4.4.0" OR (CMAKE_BUILD_TYPE STREQUAL "Debug" AND ${GCC_VERSION} VERSION_LESS "4.4.0"))
    mitkFunctionCheckCompilerFlags("-fstack-protector-all" cflags)
  endif()
  if(MINGW)
    # suppress warnings about auto imported symbols
    set(NIFTK_MITK_CXX_FLAGS "-Wl,--enable-auto-import ${NIFTK_MITK_CXX_FLAGS}")
    # we need to define a Windows version
    set(NIFTK_MITK_CXX_FLAGS "-D_WIN32_WINNT=0x0500 ${NIFTK_MITK_CXX_FLAGS}")
  endif()

  set(NIFTK_MITK_C_FLAGS "${cflags} ${NIFTK_MITK_C_FLAGS}")
  set(NIFTK_MITK_CXX_FLAGS "${cxxflags} ${NIFTK_MITK_CXX_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${NIFTK_MITK_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NIFTK_MITK_CXX_FLAGS}")
