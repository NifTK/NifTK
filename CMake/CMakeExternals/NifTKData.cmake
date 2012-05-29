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
# NifTKData - Downloads the unit-testing data as a separate project.
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTK_DATA_DIR AND NOT EXISTS ${NIFTK_DATA_DIR})
  MESSAGE(FATAL_ERROR "NIFTK_DATA_DIR variable is defined but corresponds to non-existing directory \"${NIFTK_DATA_DIR}\".")
ENDIF()

IF(BUILD_TESTING)

  SET(proj NifTKData)
  SET(proj_DEPENDENCIES )
  SET(NifTKData_DEPENDS ${proj})
 
  IF(NOT DEFINED NIFTK_DATA_DIR)

    ExternalProject_Add(${proj}
      SVN_REPOSITORY ${NIFTK_LOCATION_DATA}
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      DEPENDS ${proj_DEPENDENCIES}
    )
    
    set(NIFTK_DATA_DIR ${EP_BASE}/Source/${proj})
    MESSAGE("SuperBuild loading NifTKData from ${NIFTK_DATA_DIR}")
    
  ELSE()
  
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
    
  ENDIF(NOT DEFINED NIFTK_DATA_DIR)

ENDIF(BUILD_TESTING) 
