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

MITK_CREATE_MODULE_TESTS(LABELS niftkSurfRecon)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Suppress warning because of a bug in the Boost gil library.
  set_target_properties(${TESTDRIVER} PROPERTIES COMPILE_FLAGS "-Wno-c++11-narrowing")
endif()
