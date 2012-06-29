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
#  Original author   : j.hipwell@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#-----------------------------------------------------------------------------
# NIFTYSIM
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYSIM_ROOT AND NOT EXISTS ${NIFTYSIM_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYSIM_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYSIM_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYSIM)

  SET(proj NIFTYSIM)
  SET(proj_DEPENDENCIES VTK )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYSIM_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYSIM_ROOT)

    IF(DEFINED VTK_DIR)
      SET(USE_VTK ON)
    ELSE(DEFINED VTK_DIR)
      SET(USE_VTK OFF)
    ENDIF(DEFINED VTK_DIR)
    
    ExternalProject_Add(${proj}
      SVN_REPOSITORY ${NIFTK_LOCATION_NIFTYSIM}
      SVN_REVISION -r ${NIFTK_VERSION_NIFTYSIM}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
	      -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DCUDA_CUT_INCLUDE_DIR:PATH=${CUDA_CUT_INCLUDE_DIR}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DUSE_VIZ:BOOL=${USE_VTK}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )

    SET(NIFTYSIM_ROOT ${proj_INSTALL})
    SET(NIFTYSIM_INCLUDE_DIR "${NIFTYSIM_ROOT}/include")
    SET(NIFTYSIM_LIBRARY_DIR "${NIFTYSIM_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYSIM from ${NIFTYSIM_ROOT}")

  ELSE(NOT DEFINED NIFTYSIM_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYSIM_ROOT)

ENDIF(BUILD_NIFTYSIM)
