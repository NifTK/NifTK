project(uk_ac_ucl_cmic_newvisualization)

find_package(VL COMPONENTS VLCore VLGraphics VLVolume VLQt4 REQUIRED)
link_directories(${VL_LIBRARIES})
include_directories(${VL_INCLUDE_DIRS})

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE NEWVISUALIZATION_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS MitkCore MitkQtWidgets MitkQtWidgetsExt MitkOpenCL niftkCore 
)

  target_link_libraries(${PROJECT_NAME}
    ${VL_LIBRARIES}
    #CppMicroServices
    )