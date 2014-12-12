project(uk_ac_ucl_cmic_newvisualization)

set(_depends MitkCore MitkQtWidgets MitkQtWidgetsExt niftkCore niftkCoreGui)

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  list(APPEND _depends niftkCUDA)
endif()

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE NEWVISUALIZATION_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS ${_depends}
)

# not nice...
if(CUDA_FOUND AND NIFTK_USE_CUDA)
  include_directories(${CUDA_TOOLKIT_INCLUDE} ${CMAKE_SOURCE_DIR}/Code/Libs/CUDAKernels)
  target_link_libraries(uk_ac_ucl_cmic_newvisualization ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} niftkCUDAKernels)
endif()
