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
# NIFTYREC
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYREC_ROOT AND NOT EXISTS ${NIFTYREC_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYREC_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREC_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYREC)

  SET(proj NIFTYREC)
  SET(proj_DEPENDENCIES NIFTYREG )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYREC_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYREC_ROOT)

    ExternalProject_Add(${proj}
      SVN_REPOSITORY https://niftyrec.svn.sourceforge.net/svnroot/niftyrec/
      SVN_REVISION -r 14
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DCUDA_SDK_ROOT_DIR=${CUDA_SDK_ROOT_DIR}
      DEPENDS ${proj_DEPENDENCIES}
      )

    SET(NIFTYREC_ROOT ${proj_INSTALL})
    SET(NIFTYREC_INCLUDE_DIR "${NIFTYREC_ROOT}/include")
    SET(NIFTYREC_LIBRARY_DIR "${NIFTYREC_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYREC from ${NIFTYREC_ROOT}")

  ELSE(NOT DEFINED NIFTYREC_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYREC_ROOT)

ENDIF(BUILD_NIFTYREC)
