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

find_package(OpenMP)

set(_packages )
set(_modules MitkCore niftkOpenCV )
if(BUILD_PCL)
  list(APPEND _packages PCL)
  list(APPEND _modules niftkPCL)
endif()

MITK_CREATE_MODULE(
  DEPENDS ${_modules}
  PACKAGE_DEPENDS ${_packages}
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Suppress warning because of a bug in the Boost gil library.
  set_target_properties(niftkSurfRecon PROPERTIES COMPILE_FLAGS "-Wno-c++11-narrowing")
endif()

# only enable openmp on windows, for now.
# see https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/2502
if(MSVC)
  if(OPENMP_FOUND)
    set_target_properties(niftkSurfRecon PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkSurfRecon)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
