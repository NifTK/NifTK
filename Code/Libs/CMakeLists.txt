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

ADD_SUBDIRECTORY(Common)

IF(ITK_FOUND)
  ADD_SUBDIRECTORY(ITK)
ENDIF(ITK_FOUND)

IF(VTK_FOUND)
  ADD_SUBDIRECTORY(VTK)
ENDIF(VTK_FOUND)

IF(NVAPI_FOUND)
  ADD_SUBDIRECTORY(NVAPI)
ENDIF(NVAPI_FOUND)
