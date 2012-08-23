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

#-----------------------------------------------------------------------------
# VTK
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED VTK_DIR AND NOT EXISTS ${VTK_DIR})
  MESSAGE(FATAL_ERROR "VTK_DIR variable is defined but corresponds to non-existing directory \"${VTK_DIR}\".")
ENDIF()

SET(proj VTK)
SET(proj_DEPENDENCIES )
SET(VTK_DEPENDS ${proj})
 
IF(NOT DEFINED VTK_DIR)

  SET(additional_cmake_args )
  IF(MINGW)
    SET(additional_cmake_args
        -DCMAKE_USE_WIN32_THREADS:BOOL=ON
        -DCMAKE_USE_PTHREADS:BOOL=OFF
        -DVTK_USE_VIDEO4WINDOWS:BOOL=OFF # no header files provided by MinGW
        )
  ENDIF(MINGW)

  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_VTK}
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
        -DVTK_USE_SYSTEM_FREETYPE:BOOL=${VTK_USE_SYSTEM_FREETYPE}
        -DCMAKE_INSTALL_PREFIX:PATH=${EP_BASE}/Install/${proj}
        ${VTK_QT_ARGS}
     DEPENDS ${proj_DEPENDENCIES}
    )

  SET(VTK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading VTK from ${VTK_DIR}")

ELSE(NOT DEFINED VTK_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF(NOT DEFINED VTK_DIR)
