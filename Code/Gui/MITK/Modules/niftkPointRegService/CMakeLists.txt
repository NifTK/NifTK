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

MITK_CREATE_MODULE(niftkPointRegService
  DEPENDS niftkIGI niftkPointReg
  AUTOLOAD_WITH niftkIGI
  PACKAGE_DEPENDS
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkPointRegService)
endif()

