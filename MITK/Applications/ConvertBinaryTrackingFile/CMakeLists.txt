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
set(Qt5_LIBS)
if(MITK_USE_Qt5)
  set(Qt5_REQUIRED_COMPONENTS_BY_MODULE Core)
  find_package(Qt5 COMPONENTS ${Qt5_REQUIRED_COMPONENTS_BY_MODULE} REQUIRED QUIET)
  foreach(_component ${Qt5_REQUIRED_COMPONENTS_BY_MODULE})
    list(APPEND Qt5_LIBS ${Qt5${_component}_LIBRARIES})
  endforeach()
endif()

NIFTK_CREATE_COMMAND_LINE_APPLICATION(
  NAME niftkConvertBinaryTrackingFile
  BUILD_SLICER
  INSTALL_SCRIPT
  TARGET_LIBRARIES
    niftkcommon
    niftkVTK
    niftkIGIDataSources
    ${Qt5_LIBS}
)
