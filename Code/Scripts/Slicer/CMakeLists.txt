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

set(scripts)

if(NOT WIN32)
  set(scripts
    niftkBET.sh
    niftkCaminoMakeFAMDTRD.sh
    ${scripts}    
  )
endif()

foreach(script ${scripts})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${script}.in ${SLICER_EXECUTABLE_OUTPUTPATH}/${script} @ONLY )
  NIFTK_INSTALL_CLI(PROGRAMS ${SLICER_EXECUTABLE_OUTPUTPATH}/${script})
endforeach()


