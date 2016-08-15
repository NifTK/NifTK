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

find_package(BlackMagic)

if (BlackMagic_FOUND)

  MITK_CREATE_MODULE(
    DEPENDS niftkIGIDataSources niftkOpenCVImageConversion
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Widgets OpenCV
    AUTOLOAD_WITH niftkIGIServices
  )

  include_directories(${BlackMagic_ULTERIUS_INCLUDE_DIR})
  target_link_libraries(niftkBlackMagicDataSourceService PRIVATE ${BlackMagic_LIBRARIES})

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkBlackMagicDataSourceService)
  endif()

endif()
