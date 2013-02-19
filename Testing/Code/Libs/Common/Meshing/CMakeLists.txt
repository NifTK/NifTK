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

SET(NIFTK_MESHING_TEST_LINK_LIBRARIES 
  niftkMeshingLib
  niftkITK
  niftkVTK
  niftkcommon 
  )

SET(MESHING_UNITTEST_SRCS
  MeshGeneratorTest.cxx
)

ADD_TEST(Meshing-MeshGenerator-01 ${EXECUTABLE_OUTPUT_PATH}/MeshingUnitTests MeshGeneratorTest 1 ${INPUT_DATA}/ellipse-128-128-128-50-45-40-binary.nii.gz)
ADD_TEST(Meshing-MeshGenerator-02 ${EXECUTABLE_OUTPUT_PATH}/MeshingUnitTests MeshGeneratorTest 2 ${INPUT_DATA}/ellipse-128-128-128-50-45-40.nii 10 245)
ADD_EXECUTABLE(MeshingUnitTests MeshingUnitTests.cxx ${MESHING_UNITTEST_SRCS})
TARGET_LINK_LIBRARIES(MeshingUnitTests ${NIFTK_MESHING_TEST_LINK_LIBRARIES})
