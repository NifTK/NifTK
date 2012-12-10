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
# NIFTYREG
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYREG_ROOT AND NOT EXISTS ${NIFTYREG_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYREG_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREG_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYREG)

  SET(proj NIFTYREG)
  SET(proj_DEPENDENCIES )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYREG_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYREG_ROOT)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYREG ${NIFTK_LOCATION_NIFTYREG})

    ExternalProject_Add(${proj}
      URL ${NIFTK_LOCATION_NIFTYREG}
      URL_MD5 ${NIFTK_CHECKSUM_NIFTYREG}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_ALL_DEP:BOOL=ON
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )

  ELSE(NOT DEFINED NIFTYREG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYREG_ROOT)

ENDIF(BUILD_NIFTYREG)
