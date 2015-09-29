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

# From gcc 4.6 on, shared libaries are linked with the '--as-needed' option by default.
# This means that if a binary does not use an exported symbol from a library then that
# library will not be linked to the binary. This will cause that the static variables
# of the library will not be initialise.
# However, we use a static variable in niftkITK to register the IO factories. As a
# simple but a bit dirty workaround, here we forcibly link every library to the test
# drivers that was added to their dependencies in CMake.
if (GCC_VERSION VERSION_GREATER "4.5")
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed ${CMAKE_EXE_LINKER_FLAGS}")
endif()

add_subdirectory( Common )
add_subdirectory( RegistrationToolbox )
add_subdirectory( CorticalThickness )
add_subdirectory( BasicFilters )
add_subdirectory( BoundaryShiftIntegral )
add_subdirectory( Segmentation )
