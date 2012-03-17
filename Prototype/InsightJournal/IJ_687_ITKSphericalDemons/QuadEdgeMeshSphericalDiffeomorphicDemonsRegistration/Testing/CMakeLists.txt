SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})
SET(TEST_DATA_ROOT ${CMAKE_SOURCE_DIR}/Data)
SET(TEMP ${CMAKE_BINARY_DIR}/Testing/Temporary)

SET(QuadEdgeMeshRigidRegistration ${PROJECT_SOURCE_DIR}/../QuadEdgeMeshRigidRegistration)
SET(QuadEdgeMeshFieldSmoothingFilters ${PROJECT_SOURCE_DIR}/../QuadEdgeMeshFieldSmoothingFilters)

FIND_PATH(FREESURFER_DIR
  SetUpFreeSurfer.sh
  PATHS
  $ENV{FREESURFER_HOME}
  DOC "Directory where FreeSurfer is installed"
  )

INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/Testing
  ${QuadEdgeMeshFieldSmoothingFilters}/Source
  ${QuadEdgeMeshFieldSmoothingFilters}/Testing
  )

#######################
#######################
#
#  Define Executables
#
#######################
#######################

ADD_EXECUTABLE(itkMeshGeneratorHelperTest1 
  ${QuadEdgeMeshFieldSmoothingFilters}/Testing/itkMeshGeneratorHelperTest1.cxx )
TARGET_LINK_LIBRARIES(itkMeshGeneratorHelperTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshGeneratorHelperTest2
  ${QuadEdgeMeshFieldSmoothingFilters}/Testing/itkMeshGeneratorHelperTest2.cxx )
TARGET_LINK_LIBRARIES(itkMeshGeneratorHelperTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshGeneratorHelperTest3
  ${QuadEdgeMeshFieldSmoothingFilters}/Testing/itkMeshGeneratorHelperTest2.cxx )
TARGET_LINK_LIBRARIES(itkMeshGeneratorHelperTest3 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshGeneratorHelperTest4
  ${QuadEdgeMeshFieldSmoothingFilters}/Testing/itkMeshGeneratorHelperTest4.cxx )
TARGET_LINK_LIBRARIES(itkMeshGeneratorHelperTest4 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkMeshGeneratorHelperTest5 itkMeshGeneratorHelperTest5.cxx )
TARGET_LINK_LIBRARIES(itkMeshGeneratorHelperTest5 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1 itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1.cxx )
TARGET_LINK_LIBRARIES(itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkVersorOperationsTest1 itkVersorOperationsTest1.cxx )
TARGET_LINK_LIBRARIES(itkVersorOperationsTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkNodeScalarGradientCalculatorTest1 itkNodeScalarGradientCalculatorTest1.cxx )
TARGET_LINK_LIBRARIES(itkNodeScalarGradientCalculatorTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkNodeScalarGradientCalculatorTest2 itkNodeScalarGradientCalculatorTest2.cxx )
TARGET_LINK_LIBRARIES(itkNodeScalarGradientCalculatorTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkNodeVectorJacobianCalculatorTest1 itkNodeVectorJacobianCalculatorTest1.cxx )
TARGET_LINK_LIBRARIES(itkNodeVectorJacobianCalculatorTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkNodeVectorJacobianCalculatorTest2 itkNodeVectorJacobianCalculatorTest2.cxx )
TARGET_LINK_LIBRARIES(itkNodeVectorJacobianCalculatorTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkTriangleBasisSystemTest1 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkTriangleBasisSystemTest1.cxx )
TARGET_LINK_LIBRARIES(itkTriangleBasisSystemTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkTriangleListBasisSystemCalculatorTest1 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkTriangleListBasisSystemCalculatorTest1.cxx )
TARGET_LINK_LIBRARIES(itkTriangleListBasisSystemCalculatorTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkTriangleBasisSystemCalculatorTest1 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkTriangleBasisSystemCalculatorTest1.cxx )
TARGET_LINK_LIBRARIES(itkTriangleBasisSystemCalculatorTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkLinearInterpolateMeshFunctionTest2 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkLinearInterpolateMeshFunctionTest2.cxx )
TARGET_LINK_LIBRARIES(itkLinearInterpolateMeshFunctionTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkLinearInterpolateMeshFunctionTest1 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkLinearInterpolateMeshFunctionTest1.cxx )
TARGET_LINK_LIBRARIES(itkLinearInterpolateMeshFunctionTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkFreeSurferBinarySurfaceReaderTest1 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkFreeSurferBinarySurfaceReaderTest1.cxx )
TARGET_LINK_LIBRARIES(itkFreeSurferBinarySurfaceReaderTest1 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkFreeSurferBinarySurfaceReaderTest2 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkFreeSurferBinarySurfaceReaderTest2.cxx )
TARGET_LINK_LIBRARIES(itkFreeSurferBinarySurfaceReaderTest2 ITKQuadEdgeMesh ITKIO ITKNumerics)

ADD_EXECUTABLE(itkPointLocator2Test1 
  ${QuadEdgeMeshRigidRegistration}/Testing/itkPointLocator2Test1.cxx )
TARGET_LINK_LIBRARIES(itkPointLocator2Test1 ITKQuadEdgeMesh ITKIO ITKNumerics)


IF(USE_VTK)

  ADD_EXECUTABLE(ScalarMeshRender  ScalarMeshRender.cxx)
  TARGET_LINK_LIBRARIES(ScalarMeshRender 
    vtkRendering vtkFiltering vtkIO vtkHybrid
    ) 

  ADD_EXECUTABLE(VectorMeshRender  VectorMeshRender.cxx)
  TARGET_LINK_LIBRARIES(VectorMeshRender 
    vtkRendering vtkFiltering vtkIO vtkHybrid
    ) 

ENDIF(USE_VTK)



###################
###################
#
#  Define Tests
#
###################
###################

ADD_TEST(MeshGeneratorHelperTest1
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest1
  ${TEMP}/fixedMesh01.vtk
  ${TEMP}/movingMesh01.vtk
  )

ADD_TEST(MeshGeneratorHelperTest2
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest2
  ${TEMP}/meshWithVectors.vtk
  )

ADD_TEST(MeshGeneratorHelperTest3
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest3
  ${TEMP}/meshWithVectorsIC1.vtk
  1      # resolution
  100.0  # Scale (radius)
  )

ADD_TEST(MeshGeneratorHelperTest4
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest3
  ${TEMP}/meshWithVectorsIC4.vtk
  4      # resolution
  100.0  # Scale (radius)
  )

ADD_TEST(MeshGeneratorHelperTest5
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest3
  ${TEMP}/meshWithVectorsIC7.vtk
  7      # resolution
  100.0  # Scale (radius)
  )

ADD_TEST(MeshGeneratorHelperTest6
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest4
  ${TEMP}/meshWithScalarsIC1.vtk
  1      # resolution
  100.0  # Scale (radius)
  )

ADD_TEST(MeshGeneratorHelperTest7
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest4
  ${TEMP}/meshWithScalarsIC4.vtk
  4      # resolution
  100.0  # Scale (radius)
  )

ADD_TEST(MeshGeneratorHelperTest8
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest5
  ${TEMP}/fixedMeshWithScalarsGaussianIC4.vtk
  4      # resolution
  100.0  # Scale (radius)
  3.0    # Phi multiplier
  )

ADD_TEST(MeshGeneratorHelperTest9
  ${CXX_TEST_PATH}/itkMeshGeneratorHelperTest5
  ${TEMP}/movingMeshWithScalarsGaussianIC4.vtk
  4      # resolution
  100.0  # Scale (radius)
  2.0    # Phi multiplier
  )


ADD_TEST(DemonsFilterTest1
  ${CXX_TEST_PATH}/itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1
  ${TEMP}/meshWithScalarsIC1.vtk
  ${TEMP}/meshWithScalarsIC1.vtk
  ${TEMP}/meshWithScalarsIC1RegisteredToItself.vtk
  1.0    # Epsilon
  1.0    # SigmaX
  1.0    # Lambda
  1      # Smoothing Iterations
  1      # maximum number of iterations
  )

ADD_TEST(DemonsFilterTest2
  ${CXX_TEST_PATH}/itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1
  ${TEMP}/meshWithScalarsIC4.vtk
  ${TEMP}/meshWithScalarsIC4.vtk
  ${TEMP}/meshWithScalarsIC4RegisteredToItself.vtk
  1.0    # Epsilon
  1.0    # SigmaX
  1.0    # Lambda
  1      # Smoothing Iterations
  1      # maximum number of iterations
  )

ADD_TEST(DemonsFilterTest3
  ${CXX_TEST_PATH}/itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1
  ${TEMP}/fixedMesh01.vtk
  ${TEMP}/movingMesh01.vtk
  ${TEMP}/movingMesh01RegisteredToFixedMesh01.vtk
  1.0    # Epsilon
  1.0    # SigmaX
  1.0    # Lambda
  1      # Smoothing Iterations
  1      # maximum number of iterations
  )

ADD_TEST(DemonsFilterTest4
  ${CXX_TEST_PATH}/itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1
  ${TEST_DATA_ROOT}/Input/fixedMeshWithScalarsIC1.vtk
  ${TEST_DATA_ROOT}/Input/movingMeshWithScalarsIC1.vtk
  ${TEMP}/movingMeshWithScalarsIC1registeredToFixedMeshWithScalarsIC1.vtk
  1.0    # Epsilon
  1.0    # SigmaX
  1.0    # Lambda
  10     # Smoothing Iterations
  15     # maximum number of iterations
  )

ADD_TEST(DemonsFilterTest5
  ${CXX_TEST_PATH}/itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilterTest1
  ${TEMP}/fixedMeshWithScalarsGaussianIC4.vtk
  ${TEMP}/movingMeshWithScalarsGaussianIC4.vtk
  ${TEMP}/registeredMeshWithScalarsGaussianIC4.vtk
  0.01     # Epsilon
  100.0    # SigmaX
  1.0      # Lambda
  2        # Smoothing Iterations
  200      # maximum number of iterations
  )

ADD_TEST(VersorOperationsTest1
  ${CXX_TEST_PATH}/itkVersorOperationsTest1
  )

ADD_TEST(NodeScalarGradientCalculatorTest1
  ${CXX_TEST_PATH}/itkNodeScalarGradientCalculatorTest1
  ${TEMP}/fixedMesh01.vtk
  ${TEMP}/movingMesh01.vtk
  )

ADD_TEST(NodeScalarGradientCalculatorTest2
  ${CXX_TEST_PATH}/itkNodeScalarGradientCalculatorTest2
  )

ADD_TEST(NodeVectorJacobianCalculatorTest1
  ${CXX_TEST_PATH}/itkNodeVectorJacobianCalculatorTest1
  ${TEMP}/fixedMesh01.vtk
  ${TEMP}/movingMesh01.vtk
  )

ADD_TEST(NodeVectorJacobianCalculatorTest2
  ${CXX_TEST_PATH}/itkNodeVectorJacobianCalculatorTest2
  )

ADD_TEST(TriangleBasisSystemTest1
  ${CXX_TEST_PATH}/itkTriangleBasisSystemTest1
 )

ADD_TEST(TriangleListBasisSystemCalculatorTest1
  ${CXX_TEST_PATH}/itkTriangleListBasisSystemCalculatorTest1
 )

ADD_TEST(TriangleBasisSystemCalculatorTest1
  ${CXX_TEST_PATH}/itkTriangleBasisSystemCalculatorTest1
 )

ADD_TEST(PointLocator2Test1
  ${CXX_TEST_PATH}/itkPointLocator2Test1
  ${TEMP}/itkFreeSurferBinarySurfaceReaderTest2.vtk
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

ADD_TEST(itkLinearInterpolateMeshFunctionTest2
  ${CXX_TEST_PATH}/itkLinearInterpolateMeshFunctionTest2
  ${TEMP}
  )

ADD_TEST(itkLinearInterpolateMeshFunctionTest1
  ${CXX_TEST_PATH}/itkLinearInterpolateMeshFunctionTest1
  ${FREESURFER_DIR}/subjects/bert/surf/lh.pial
  ${FREESURFER_DIR}/subjects/bert/surf/lh.curv.pial
  )

ENDIF( FREESURFER_DIR )

ADD_TEST(itkSimpleSurface1
  ${CXX_TEST_PATH}/itkFreeSurferBinarySurfaceReaderTest2
  ${TEST_DATA_ROOT}/Input/SimpleSurface1/lh.sphere
  ${TEST_DATA_ROOT}/Input/SimpleSurface1/lh.sulc
  ${TEMP}/itkSimpleSurface1.vtk
  )

ADD_TEST(itkSimpleSurface2
  ${CXX_TEST_PATH}/itkFreeSurferBinarySurfaceReaderTest2
  ${TEST_DATA_ROOT}/Input/SimpleSurface2/lh.sphere
  ${TEST_DATA_ROOT}/Input/SimpleSurface2/lh.sulc
  ${TEMP}/itkSimpleSurface2.vtk
  )

IF(USE_VTK)
  IF(GENERATE_PAPER_SCREENSHOTS)

    ADD_TEST(ScalarRenderTest1
      ${CXX_TEST_PATH}/ScalarMeshRender
      ${TEMP}/fixedMeshWithScalarsGaussianIC4.vtk
      ${TEMP}/fixedMeshWithScalarsGaussianIC4.png
      )

    ADD_TEST(ScalarRenderTest2
      ${CXX_TEST_PATH}/ScalarMeshRender
      ${TEMP}/movingMeshWithScalarsGaussianIC4.vtk
      ${TEMP}/movingMeshWithScalarsGaussianIC4.png
      )

    ADD_TEST(ScalarRenderTest3
      ${CXX_TEST_PATH}/ScalarMeshRender
      ${TEMP}/registeredMeshWithScalarsGaussianIC4.vtk
      ${TEMP}/registeredMeshWithScalarsGaussianIC4.png
      )


    ADD_TEST(VectorRenderTest1
      ${CXX_TEST_PATH}/VectorMeshRender
      ${TEMP}/meshWithVectorsIC4g.vtk
      1   # vector scale factor
      ${TEMP}/meshWithVectorsIC4g.png
      )

  ENDIF(GENERATE_PAPER_SCREENSHOTS)
ENDIF(USE_VTK)

