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

MITK_CREATE_MODULE(niftkNVidia
  EXPORT_DEFINE NIFTKNVIDIA_EXPORT
  INCLUDE_DIRS DataSources
  DEPENDS niftkIGI
)

target_link_libraries(niftkNVidia
  niftknvapi
  )

if(BUILD_TESTING)  
  add_subdirectory(Testing)
endif()
