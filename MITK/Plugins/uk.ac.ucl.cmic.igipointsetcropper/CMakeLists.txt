project(uk_ac_ucl_cmic_igipointsetcropper)

mitk_create_plugin(
  EXPORT_DIRECTIVE NIFTK_IMAGECROPPER
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS MitkQtWidgetsExt
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(uk_ac_ucl_cmic_igipointsetcropper)
endif()
