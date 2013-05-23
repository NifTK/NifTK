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


#-----------------------------------------------------------------------------
# VTK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED VTK_DIR AND NOT EXISTS ${VTK_DIR})
  message(FATAL_ERROR "VTK_DIR variable is defined but corresponds to non-existing directory \"${VTK_DIR}\".")
endif()

set(proj VTK)
set(proj_DEPENDENCIES )
set(VTK_DEPENDS ${proj})

if(NOT DEFINED VTK_DIR)

  set(additional_cmake_args )
  if(MINGW)
    set(additional_cmake_args
        -DCMAKE_USE_WIN32_THREADS:BOOL=ON
        -DCMAKE_USE_PTHREADS:BOOL=OFF
        -DVTK_USE_VIDEO4WINDOWS:BOOL=OFF # no header files provided by MinGW
        )
  endif(MINGW)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_VTK ${NIFTK_LOCATION_VTK})

  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_VTK}
    URL_MD5 ${NIFTK_CHECKSUM_VTK}
    BINARY_DIR ${proj}-build
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${additional_cmake_args}
        -DDESIRED_QT_VERSION:STRING=4
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DVTK_WRAP_TCL:BOOL=OFF
        -DVTK_WRAP_PYTHON:BOOL=OFF
        -DVTK_WRAP_JAVA:BOOL=OFF
        -DVTK_USE_RPATH:BOOL=ON
        -DVTK_USE_PARALLEL:BOOL=ON
        -DVTK_USE_CHARTS:BOOL=OFF
        -DVTK_USE_QTCHARTS:BOOL=ON
        -DVTK_USE_GEOVIS:BOOL=OFF
        -DVTK_LEGACY_REMOVE:BOOL=ON
        -DVTK_USE_SYSTEM_FREETYPE:BOOL=${VTK_USE_SYSTEM_FREETYPE}
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
        ${VTK_QT_ARGS}
     DEPENDS ${proj_DEPENDENCIES}
    )

  set(VTK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  message("SuperBuild loading VTK from ${VTK_DIR}")

else(NOT DEFINED VTK_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED VTK_DIR)
