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

if (UltrasonixUlterius_FOUND)

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources niftkQImageConversion
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Widgets  NiftyLink|OpenIGTLink
    AUTOLOAD_WITH niftkIGIDataSources
  )

  include_directories(${Ultrasonix_ULTERIUS_INCLUDE_DIR})
  target_link_libraries(niftkUltrasonixDataSourceService PRIVATE ${Ultrasonix_ULTERIUS_LIBRARY})

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkUltrasonixDataSourceService)
  endif()

endif()
