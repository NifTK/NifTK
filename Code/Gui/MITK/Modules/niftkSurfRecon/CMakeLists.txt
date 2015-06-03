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
if(BUILD_PCL)
  list(APPEND _packages PCL)
endif()

MITK_CREATE_MODULE(niftkSurfRecon
  EXPORT_DEFINE NIFTKSURFRECON_EXPORT
  DEPENDS MitkCore niftkOpenCV niftkPCL
  PACKAGE_DEPENDS ${_packages}
)

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
