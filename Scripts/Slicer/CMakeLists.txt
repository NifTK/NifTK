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
    niftkBET
    ${scripts}    
  )
endif()

foreach(script ${scripts})
  NIFTK_CREATE_COMMAND_LINE_SCRIPT(NAME ${script} EXTENSION sh INSTALL_CLI_MODULES)
endforeach()


