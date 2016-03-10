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
  DEPENDS niftkCore niftkIGIDataSources niftkIGITrackers
  PACKAGE_DEPENDS CTK|CTKWidgets
  AUTOLOAD_WITH niftkIGIServices
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkMITKTrackerDataSourceService)
endif()

