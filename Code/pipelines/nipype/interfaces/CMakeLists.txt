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

## #################################################################
## Installation of the niftk python module in the install bin path
## #################################################################

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/niftk 
  DESTINATION python 
  USE_SOURCE_PERMISSIONS
  )
