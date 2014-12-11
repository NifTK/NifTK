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
set(python_scripts)
set(python_scripts_no_cli)

if(NOT WIN32)
  set(scripts
    niftkFillLesions
    ${scripts}    
  )
  foreach(script ${scripts})
    NIFTK_CREATE_COMMAND_LINE_SCRIPT(NAME ${script} EXTENSION sh INSTALL_CLI_MODULES)
  endforeach()
endif()

set(python_scripts
    niftkBETter
    niftkBiasFieldCorrection
    niftkAtrophyCalculator
    niftkRegMNI-T1-B0
    ${python_scripts}
)
foreach(script ${python_scripts})
  NIFTK_CREATE_COMMAND_LINE_SCRIPT(NAME ${script} EXTENSION py INSTALL_CLI_MODULES)
endforeach()

set(python_scripts_no_cli
    niftkT1PDT2Lesions
    niftkMTPFillLesions
    ${python_scripts_no_cli}
)
foreach(script ${python_scripts_no_cli})
  NIFTK_CREATE_COMMAND_LINE_SCRIPT(NAME ${script} EXTENSION py )
endforeach()

