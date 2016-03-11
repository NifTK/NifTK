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

##############################################################################
# Note: As a 'Service Only' Module, this module should only depend on
# niftkIGIServices and niftkICPReg. It should not be necessary to add
# any other module dependencies. Extra dependencies could/should be added
# to niftkICPReg.
##############################################################################

MITK_CREATE_MODULE(
  DEPENDS niftkIGIServices niftkICPReg
  AUTOLOAD_WITH niftkIGIServices
  PACKAGE_DEPENDS
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkICPRegService)
endif()

