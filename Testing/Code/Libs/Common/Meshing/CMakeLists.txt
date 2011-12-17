#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-09-26 13:46:27 +0100 (Mon, 26 Sep 2011) $
#  Revision          : $LastChangedRevision: 7365 $
#  Last modified by  : $LastChangedBy: sj $
#
#  Original author   : stian.johnsen.09@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
SET(NIFTK_MESHING_TEST_LINK_LIBRARIES 
  niftkMeshingLib
  niftkITK
  niftkITKVTK
  niftkcommon 
  )

SET(MESHING_UNITTEST_SRCS
  MeshGeneratorTest.cxx
)

ADD_TEST(Meshing-MeshGenerator-01 ${EXECUTABLE_OUTPUT_PATH}/MeshingUnitTests MeshGeneratorTest 1 ${INPUT_DATA}/ellipse-128-128-128-50-45-40-binary.nii.gz)
ADD_TEST(Meshing-MeshGenerator-02 ${EXECUTABLE_OUTPUT_PATH}/MeshingUnitTests MeshGeneratorTest 2 ${INPUT_DATA}/ellipse-128-128-128-50-45-40.nii 10 245)
ADD_EXECUTABLE(MeshingUnitTests MeshingUnitTests.cxx ${MESHING_UNITTEST_SRCS})
TARGET_LINK_LIBRARIES(MeshingUnitTests ${NIFTK_MESHING_TEST_LINK_LIBRARIES})
