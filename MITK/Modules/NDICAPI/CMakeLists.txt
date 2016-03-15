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
  INCLUDE_DIRS ndicapi
  NO_INIT      # Not an MITK MicroServices module
  FORCE_STATIC # As it doesn't use the MITK exports 'standard'.
)

if(NOT WIN32)
  set_target_properties( niftkNDICAPI PROPERTIES COMPILE_FLAGS "-fPIC")
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkNDICAPI)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

