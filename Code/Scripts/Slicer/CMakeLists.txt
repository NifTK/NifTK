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

SET(scripts)

IF(NOT WIN32)
  SET(scripts
    niftkBET.sh
    niftkCaminoMakeFAMDTRD.sh
    ${scripts}    
  )
ENDIF()

FOREACH(script ${scripts})
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${script}.in ${SLICER_EXECUTABLE_OUTPUTPATH}/${script} @ONLY )
  NIFTK_INSTALL_CLI(PROGRAMS ${SLICER_EXECUTABLE_OUTPUTPATH}/${script})
ENDFOREACH()


