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
if(MITK_USE_OpenCV)
  find_package(OpenCV REQUIRED)
endif()

find_package(Caffe REQUIRED)
if (Caffe_FOUND)

  include_directories(${Caffe_INCLUDE_DIRS})
  list(APPEND ALL_INCLUDE_DIRECTORIES ${Caffe_INCLUDE_DIRS})
  list(APPEND ALL_LIBRARIES ${Caffe_LIBRARIES})
  add_definitions(${Caffe_DEFINITIONS})
 
endif()
