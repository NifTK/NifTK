PROJECT(uk_ac_ucl_cmic_xnat)

set(QT_USE_QTNETWORK TRUE)

# To process the result of the queries in XML format 
#set(QT_USE_QTXML TRUE)

# To process the result of the queries in json format 
set(QT_USE_QTSCRIPT TRUE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/internal)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE XNAT_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES Mitk MitkExt Qmitk QmitkExt XnatRestWidgets CTK
)
