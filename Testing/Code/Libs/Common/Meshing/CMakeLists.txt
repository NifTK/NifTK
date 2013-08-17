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

set(NIFTK_MESHING_TEST_LINK_LIBRARIES 
  niftkMeshingLib
  niftkITK
  niftkVTK
  niftkcommon 
  )

set(MESHING_UNITTEST_SRCS
  niftkMeshGeneratorTest.cxx
)

add_test(Meshing-MeshGenerator-01 ${EXECUTABLE_OUTPUT_PATH}/niftkMeshingUnitTests niftkMeshGeneratorTest 1 ${INPUT_DATA}/ellipse-128-128-128-50-45-40-binary.nii.gz)
add_test(Meshing-MeshGenerator-02 ${EXECUTABLE_OUTPUT_PATH}/niftkMeshingUnitTests niftkMeshGeneratorTest 2 ${INPUT_DATA}/ellipse-128-128-128-50-45-40.nii 10 245)
add_executable(niftkMeshingUnitTests niftkMeshingUnitTests.cxx ${MESHING_UNITTEST_SRCS})
target_link_libraries(niftkMeshingUnitTests ${NIFTK_MESHING_TEST_LINK_LIBRARIES})
