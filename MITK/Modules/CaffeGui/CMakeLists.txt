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
  DEPENDS MitkQtWidgets niftkCaffe niftkCoreGui
  PACKAGE_DEPENDS OpenCV Caffe
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCaffeGui)
endif()
