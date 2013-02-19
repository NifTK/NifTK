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

ADD_SUBDIRECTORY(Libs)

IF(BUILD_COMMAND_LINE_PROGRAMS)
  ADD_SUBDIRECTORY(Applications)
ENDIF()

IF(BUILD_COMMAND_LINE_SCRIPTS)
  ADD_SUBDIRECTORY(Scripts)
ENDIF()

IF(BUILD_GUI)
  ADD_SUBDIRECTORY(Gui)
ENDIF(BUILD_GUI)

