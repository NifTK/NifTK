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

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

NIFTK_CREATE_COMMAND_LINE_APPLICATION(
  NAME niftkCaffeSeg
  BUILD_SLICER
  INSTALL_SCRIPT
  TARGET_LIBRARIES
    niftkCore
    niftkCaffe
    ${Caffe_LIBRARIES}
)
