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

if(NOT WIN32)
  if(APPLE)
    configure_file(${CMAKE_SOURCE_DIR}/Scripts/Matlab/matlab_shell-For-Mac.sh.in ${EXECUTABLE_OUTPUT_PATH}/matlab_shell.sh @ONLY)
  else()
    configure_file(${CMAKE_SOURCE_DIR}/Scripts/Matlab/matlab_shell.sh.in ${EXECUTABLE_OUTPUT_PATH}/matlab_shell.sh @ONLY)
  endif()
  MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/matlab_shell.sh)
endif()

