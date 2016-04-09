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

find_package(UltrasonixUlterius)

if (UltrasonixUlterius_FOUND)

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources niftkOpenCVImageConversion
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Widgets OpenCV
    AUTOLOAD_WITH niftkIGI
  )

  include_directories(${Ultrasonix_ULTERIUS_INCLUDE_DIR})
  target_link_libraries(niftkUltrasonixDataSourceService PRIVATE ${Ultrasonix_ULTERIUS_LIBRARY})

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkUltrasonixDataSourceService)
  endif()

endif()
