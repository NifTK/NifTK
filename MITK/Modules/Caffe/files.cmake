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

set(CPP_FILES
  Segmentation/niftkCaffeFCNSegmentor.cxx
)
if(WIN32)
  list(APPEND CPP_FILES ${Caffe_SOURCE_DIR}/src/caffe/util/force_link.cxx)
endif()


