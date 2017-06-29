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

if (Atracsys_FOUND)

  MITK_CREATE_MODULE(
    DEPENDS niftkAtracsys niftkIGIDataSources 
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Core NiftyLink|OpenIGTLink CTK|CTKWidgets
    AUTOLOAD_WITH niftkIGIDataSources
  )

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkAtracsysDataSourceService)
  endif()

endif()
