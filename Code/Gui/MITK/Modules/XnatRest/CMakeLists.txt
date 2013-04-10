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

MITK_CREATE_MODULE(XnatRest
  EXPORT_DEFINE XnatRest_EXPORT
  PACKAGE_DEPENDS curl zlib
)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(XnatRest PROPERTIES COMPILE_FLAGS "-fvisibility=default")
endif()

IF(BUILD_TESTING)  
  ADD_SUBDIRECTORY(Testing)
ENDIF()  
