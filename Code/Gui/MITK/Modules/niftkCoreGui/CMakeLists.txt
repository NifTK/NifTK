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

set(_modules MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore)
if(BUILD_VL)
  set(_modules
    ${_modules}
    MitkOpenCL
  )
endif()

MITK_CREATE_MODULE(niftkCoreGui
  EXPORT_DEFINE NIFTKCOREGUI_EXPORT
  INCLUDE_DIRS Events LookupTables
  DEPENDS ${_modules}
  PACKAGE_DEPENDS Qt4|QtGui CTK|CTKWidgets
)

target_link_libraries(niftkCoreGui
  niftkITK
  niftkVTK
)

if(BUILD_VL)
  target_link_libraries(niftkCoreGui
    ${VL_LIBRARIES}
  )
endif()


if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCoreGui)
endif()

if(BUILD_TESTING)
  #add_subdirectory(Testing)
endif(BUILD_TESTING)
