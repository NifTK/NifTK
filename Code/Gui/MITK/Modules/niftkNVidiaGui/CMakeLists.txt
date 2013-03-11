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

MITK_CREATE_MODULE(niftkNVidiaGui
  EXPORT_DEFINE NIFTKNVIDIAGUI_EXPORT
  INCLUDE_DIRS ToolsGui
  DEPENDS niftkIGIGui niftkNVidia
  PACKAGE_DEPENDS QT
  QT_MODULE 
)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/video)
INCLUDE_DIRECTORIES(${QT_QTOPENGL_INCLUDE_DIR})

# stuffing this into DEPENDS above doesnt work for some reason
TARGET_LINK_LIBRARIES(niftkNVidiaGui
  niftknvapi
  )
  
IF(BUILD_TESTING)  
  ADD_SUBDIRECTORY(Testing)
ENDIF()
