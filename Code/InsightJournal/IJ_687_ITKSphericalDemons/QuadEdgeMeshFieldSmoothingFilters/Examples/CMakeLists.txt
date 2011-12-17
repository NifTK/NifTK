#
# Add here command lines to build examples.
#

INCLUDE_DIRECTORIES( ${QuadEdgeMeshFieldSmoothingFilters_SOURCE_DIR}/Testing )

ADD_EXECUTABLE(SmoothingMeshWithScalars SmoothingMeshWithScalars.cxx) 
TARGET_LINK_LIBRARIES(SmoothingMeshWithScalars ITKQuadEdgeMesh ITKNumerics ITKIO) 

ADD_EXECUTABLE(SmoothingMeshWithVectors SmoothingMeshWithVectors.cxx) 
TARGET_LINK_LIBRARIES(SmoothingMeshWithVectors ITKQuadEdgeMesh ITKNumerics ITKIO) 

