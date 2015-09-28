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
  INCLUDE_DIRS 
  DEPENDS niftkCore MitkSceneSerializationBase
  AUTOLOAD_WITH niftkCore
)

target_link_libraries(niftkCoreIO PRIVATE niftkITK)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCoreIO)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
