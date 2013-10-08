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


# Additional stuff that gets included if we are using MITK

include(mitkFunctionCheckCompilerFlags)
include(mitkFunctionGetGccVersion)

if(BUILD_GUI)
  message("Qt version=${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}.${QT_VERSION_PATCH}")
  message("CTK version=${NIFTK_VERSION_CTK}")
endif()
if(BUILD_IGI)
  message("NiftyLink version=${NIFTK_VERSION_NIFTYLINK}")
endif(BUILD_IGI)

# Print out other versions.
message("Boost version=${NIFTK_VERSION_Boost}")
message("GDCM version=${NIFTK_VERSION_GDCM}")
message("DCMTK version=${NIFTK_VERSION_DCMTK}")
message("ITK version=${NIFTK_VERSION_ITK}")
message("VTK version=${NIFTK_VERSION_VTK}")
message("MITK version=${NIFTK_VERSION_MITK}")

# MinGW does not export all symbols automatically, so no need to set flags
if(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW)
  #set(VISIBILITY_CXX_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden")
endif(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW)

if(NOT UNIX AND NOT MINGW)
  set(MITK_WIN32_FORCE_STATIC "STATIC")
endif(NOT UNIX AND NOT MINGW)

if(MITK_USE_QT)
  set(QT_QMAKE_EXECUTABLE ${MITK_QMAKE_EXECUTABLE})
  add_definitions(-DQWT_DLL)
endif()

set(NIFTK_MITK_C_FLAGS "${NIFTK_COVERAGE_C_FLAGS} ${NIFTK_ADDITIONAL_C_FLAGS}")
set(NIFTK_MITK_CXX_FLAGS "${VISIBILITY_CXX_FLAGS} ${NIFTK_COVERAGE_CXX_FLAGS} ${NIFTK_ADDITIONAL_CXX_FLAGS}")

if(APPLE)
  set(NIFTK_MITK_CXX_FLAGS "${NIFTK_MITK_CXX_FLAGS} -DNIFTK_OS_IS_MAC")
endif(APPLE)

if(CMAKE_COMPILER_IS_GNUCXX)

  set(MITK_GNU_COMPILER_C_WARNINGS "-Wall -Wextra -Wpointer-arith -Winvalid-pch -Wcast-align -Wwrite-strings -D_FORTIFY_SOURCE=2")
  set(MITK_GNU_COMPILER_CXX_WARNINGS "-Woverloaded-virtual -Wold-style-cast -Wstrict-null-sentinel -Wsign-promo ")

  if(NIFTK_VERBOSE_COMPILER_WARNINGS)
    set(cflags "${MITK_GNU_COMPILER_C_WARNINGS}")
    set(cxxflags "${cflags} ${MITK_GNU_COMPILER_CXX_WARNINGS}") 
  endif(NIFTK_VERBOSE_COMPILER_WARNINGS)
  
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

set(${PROJECT_NAME}_MODULES_PACKAGE_DEPENDS_DIR "${PROJECT_SOURCE_DIR}/CMake/PackageDepends")
list(APPEND MODULES_PACKAGE_DEPENDS_DIRS ${${PROJECT_NAME}_MODULES_PACKAGE_DEPENDS_DIR})
