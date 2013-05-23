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

add_subdirectory(Libs)

if(BUILD_COMMAND_LINE_PROGRAMS)
  add_subdirectory(Applications)
endif()

if(BUILD_COMMAND_LINE_SCRIPTS)
  add_subdirectory(Scripts)
endif()

if(BUILD_GUI)
  add_subdirectory(Gui)
endif(BUILD_GUI)

