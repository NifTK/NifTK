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
  PACKAGE_DEPENDS Qt4|QtCore Qt5|Core
  NO_INIT      # Not an MITK MicroServices module
)

if(WIN32)
  target_link_libraries(niftkSerialPort PRIVATE Setupapi)
endif()

if(BUILD_SHARED_LIBS)
  add_definitions(-DQEXTSERIALPORT_BUILD_SHARED)
endif()

if(BUILD_SHARED_LIBS)
  add_definitions(-DQEXTSERIALPORT_BUILD_SHARED)
endif()

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-framework CoreFoundation -framework IOKit ")
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkSerialPort)
endif()

