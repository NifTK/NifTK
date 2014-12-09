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
