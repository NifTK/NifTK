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

MITK_CREATE_MODULE(
  DEPENDS niftkCore MitkSceneSerializationBase
  PACKAGE_DEPENDS Qt4|QtCore Qt5|Core Qt5|Gui Qt5|Xml
  AUTOLOAD_WITH niftkCore
)

target_link_libraries(niftkCoreIO PRIVATE niftkITKIO)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCoreIO)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
