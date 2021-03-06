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

add_subdirectory(Common)
add_subdirectory(Freesurfer)
add_subdirectory(Fluid)
add_subdirectory(Registration)
add_subdirectory(SIENA)
add_subdirectory(GradwarpCorrection)
add_subdirectory(BoundaryShiftIntegral)
add_subdirectory(TBSS)
add_subdirectory(Utilities)
add_subdirectory(CorticalThickness)
add_subdirectory(Parcellation)
add_subdirectory(MAPS)
add_subdirectory(Slicer)
add_subdirectory(IGI)
add_subdirectory(Matlab)

if(NOT WIN32)
  add_subdirectory(MSTools)
endif()

if(BUILD_CAMINO)
  add_subdirectory(Camino)
endif()
