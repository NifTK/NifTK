FIND_PATH(FREESURFER_DIR
  SetUpFreeSurfer.sh
  PATHS
  $ENV{FREESURFER_HOME}
  DOC "Directory where FreeSurfer is installed"
  )

#######################
#######################
#
#  Define Executables
#
#######################
#######################

ADD_EXECUTABLE(itkTriangleListBasisSystemCalculatorTest1 itkTriangleListBasisSystemCalculatorTest1.cxx )
TARGET_LINK_LIBRARIES(itkTriangleListBasisSystemCalculatorTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkTriangleBasisSystemCalculatorTest1 itkTriangleBasisSystemCalculatorTest1.cxx )
TARGET_LINK_LIBRARIES(itkTriangleBasisSystemCalculatorTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkTriangleBasisSystemTest1 itkTriangleBasisSystemTest1.cxx )
TARGET_LINK_LIBRARIES(itkTriangleBasisSystemTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkRegularSphereQuadEdgeMeshSourceTest1 itkRegularSphereQuadEdgeMeshSourceTest1.cxx )
TARGET_LINK_LIBRARIES(itkRegularSphereQuadEdgeMeshSourceTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkFreeSurferBinarySurfaceReaderTest1 itkFreeSurferBinarySurfaceReaderTest1.cxx )
TARGET_LINK_LIBRARIES(itkFreeSurferBinarySurfaceReaderTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkFreeSurferBinarySurfaceReaderTest2 itkFreeSurferBinarySurfaceReaderTest2.cxx )
TARGET_LINK_LIBRARIES(itkFreeSurferBinarySurfaceReaderTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshFunctionTest1 itkMeshFunctionTest1.cxx )
TARGET_LINK_LIBRARIES(itkMeshFunctionTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkLinearInterpolateMeshFunctionTest3 itkLinearInterpolateMeshFunctionTest3.cxx )
TARGET_LINK_LIBRARIES(itkLinearInterpolateMeshFunctionTest3 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkLinearInterpolateMeshFunctionTest2 itkLinearInterpolateMeshFunctionTest2.cxx )
TARGET_LINK_LIBRARIES(itkLinearInterpolateMeshFunctionTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkLinearInterpolateMeshFunctionTest1 itkLinearInterpolateMeshFunctionTest1.cxx )
TARGET_LINK_LIBRARIES(itkLinearInterpolateMeshFunctionTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkNearestNeighborInterpolateMeshFunctionTest1 itkNearestNeighborInterpolateMeshFunctionTest1.cxx )
TARGET_LINK_LIBRARIES(itkNearestNeighborInterpolateMeshFunctionTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkQuadEdgeMeshKdTreeGeneratorTest itkQuadEdgeMeshKdTreeGeneratorTest.cxx )
TARGET_LINK_LIBRARIES(itkQuadEdgeMeshKdTreeGeneratorTest ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkPointLocator2Test1 itkPointLocator2Test1.cxx )
TARGET_LINK_LIBRARIES(itkPointLocator2Test1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshToMeshRegistrationMethodTest1 itkMeshToMeshRegistrationMethodTest1.cxx)
TARGET_LINK_LIBRARIES(itkMeshToMeshRegistrationMethodTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshToMeshRegistrationMethodTest2 itkMeshToMeshRegistrationMethodTest2.cxx)
TARGET_LINK_LIBRARIES(itkMeshToMeshRegistrationMethodTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshToMeshRegistrationMethodTest3 itkMeshToMeshRegistrationMethodTest3.cxx)
TARGET_LINK_LIBRARIES(itkMeshToMeshRegistrationMethodTest3 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshToMeshMetricTest1 itkMeshToMeshMetricTest1.cxx)
TARGET_LINK_LIBRARIES(itkMeshToMeshMetricTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeanSquaresMeshToMeshMetricTest1 itkMeanSquaresMeshToMeshMetricTest1.cxx)
TARGET_LINK_LIBRARIES(itkMeanSquaresMeshToMeshMetricTest1 ITKQuadEdgeMesh ITKNumerics ITKIO)

ADD_EXECUTABLE(itkMeanSquaresMeshToMeshMetricTest2 itkMeanSquaresMeshToMeshMetricTest2.cxx)
TARGET_LINK_LIBRARIES(itkMeanSquaresMeshToMeshMetricTest2 ITKQuadEdgeMesh ITKNumerics ITKIO)

ADD_EXECUTABLE(itkMeanSquaresMeshToMeshMetricTest3 itkMeanSquaresMeshToMeshMetricTest3.cxx)
TARGET_LINK_LIBRARIES(itkMeanSquaresMeshToMeshMetricTest3 ITKQuadEdgeMesh ITKNumerics ITKIO)

ADD_EXECUTABLE(itkImageToImageRotationRegistrationTest itkImageToImageRotationRegistrationTest.cxx)
TARGET_LINK_LIBRARIES(itkImageToImageRotationRegistrationTest ITKNumerics ITKIO)


###################
###################
#
#  Define Tests
#
###################
###################

ADD_TEST(itkTriangleListBasisSystemCalculatorTest1
  ${CXX_TEST_PATH}/itkTriangleListBasisSystemCalculatorTest1
 )

ADD_TEST(itkTriangleBasisSystemCalculatorTest1
  ${CXX_TEST_PATH}/itkTriangleBasisSystemCalculatorTest1
 )

ADD_TEST(itkTriangleBasisSystemTest1
  ${CXX_TEST_PATH}/itkTriangleBasisSystemTest1
 )

ADD_TEST(itkRegularSphereQuadEdgeMeshSourceTest1
  ${CXX_TEST_PATH}/itkRegularSphereQuadEdgeMeshSourceTest1
  ${TEMP}/itkRegularSphereQuadEdgeMeshSourceTest1.vtk
 )

ADD_TEST(itkMeanSquaresMeshToMeshMetricTest1
  ${CXX_TEST_PATH}/itkMeanSquaresMeshToMeshMetricTest1
  )

ADD_TEST(itkMeanSquaresMeshToMeshMetricTest2
  ${CXX_TEST_PATH}/itkMeanSquaresMeshToMeshMetricTest2
  )

ADD_TEST(itkMeanSquaresMeshToMeshMetricTest3
  ${CXX_TEST_PATH}/itkMeanSquaresMeshToMeshMetricTest3
  )


IF( FREESURFER_DIR )

ADD_TEST(itkFreeSurferBinarySurfaceReaderTest1
  ${CXX_TEST_PATH}/itkFreeSurferBinarySurfaceReaderTest1
  ${FREESURFER_DIR}/subjects/bert/surf/lh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/lh.curv.pial
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest1.vtk
  )

ADD_TEST(itkFreeSurferBinarySurfaceReaderTest2
  ${CXX_TEST_PATH}/itkFreeSurferBinarySurfaceReaderTest2
  ${FREESURFER_DIR}/subjects/bert/surf/lh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/lh.curv.pial
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest2.vtk
  )

ADD_TEST(itkFreeSurferBinarySurfaceReaderTest3
  ${CXX_TEST_PATH}/itkFreeSurferBinarySurfaceReaderTest1
  ${FREESURFER_DIR}/subjects/bert/surf/rh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/rh.curv.pial
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest3.vtk
  )

ADD_TEST(itkFreeSurferBinarySurfaceReaderTest4
  ${CXX_TEST_PATH}/itkFreeSurferBinarySurfaceReaderTest2
  ${FREESURFER_DIR}/subjects/bert/surf/rh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/rh.curv.pial
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest4.vtk
  )

ADD_TEST(itkMeshFunctionTest1
  ${CXX_TEST_PATH}/itkMeshFunctionTest1
  )

ADD_TEST(itkLinearInterpolateMeshFunctionTest2
  ${CXX_TEST_PATH}/itkLinearInterpolateMeshFunctionTest2
  ${TEMP}
  )

ADD_TEST(itkLinearInterpolateMeshFunctionTest3
  ${CXX_TEST_PATH}/itkLinearInterpolateMeshFunctionTest3
  ${TEMP}
  )

ADD_TEST(itkLinearInterpolateMeshFunctionTest1
  ${CXX_TEST_PATH}/itkLinearInterpolateMeshFunctionTest1
  ${FREESURFER_DIR}/subjects/bert/surf/lh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/lh.curv.pial
  )

ADD_TEST(itkNearestNeighborInterpolateMeshFunctionTest1
  ${CXX_TEST_PATH}/itkNearestNeighborInterpolateMeshFunctionTest1
  ${FREESURFER_DIR}/subjects/bert/surf/lh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/lh.curv.pial
  )

ADD_TEST(itkQuadEdgeMeshKdTreeGeneratorTest1
  ${CXX_TEST_PATH}/itkQuadEdgeMeshKdTreeGeneratorTest
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest2.vtk
  ${TEMP}/itkQuadEdgeMeshKdTreeGeneratorTest1.vtk
  )

ADD_TEST(itkPointLocator2Test1
  ${CXX_TEST_PATH}/itkPointLocator2Test1
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest2.vtk
  )

ADD_TEST(itkMeshToMeshRegistrationMethodTest1a
  ${CXX_TEST_PATH}/itkMeshToMeshRegistrationMethodTest1
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest1.vtk
  )

ADD_TEST(itkMeshToMeshRegistrationMethodTest2
  ${CXX_TEST_PATH}/itkMeshToMeshRegistrationMethodTest2
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest1.vtk
  )

ADD_TEST(itkMeshToMeshRegistrationMethodTest3
  ${CXX_TEST_PATH}/itkMeshToMeshRegistrationMethodTest3
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest1.vtk
  )

ADD_TEST(itkMeshToMeshMetricTest1a
  ${CXX_TEST_PATH}/itkMeshToMeshMetricTest1
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest1.vtk
  )


ENDIF( FREESURFER_DIR )

