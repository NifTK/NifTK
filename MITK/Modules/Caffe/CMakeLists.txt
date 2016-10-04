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

set(_modules niftkOpenCVImageConversion)

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  list(APPEND _modules niftkCUDA)
endif()

MITK_CREATE_MODULE(
  INCLUDE_DIRS Segmentation
  DEPENDS ${_modules}
  PACKAGE_DEPENDS OpenCV Caffe
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCaffe)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
