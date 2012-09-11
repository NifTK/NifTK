project(uk_ac_ucl_cmic_niftyreg)

INCLUDE_DIRECTORIES(${NIFTYREG_DIR}/include)
LINK_DIRECTORIES(${NIFTYREG_DIR}/lib)
ADD_DEFINITIONS(-DUSE_NIFTYREG)

MACRO_CREATE_MITK_CTK_PLUGIN(
  EXPORT_DIRECTIVE NIFTYREG_EXPORTS
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDENCIES CTK QmitkExt niftkMitkExt niftkQmitkExt
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                      niftkITK
		      _reg_KLdivergence
		      _reg_blockMatching
		      _reg_femTransformation
		      _reg_globalTransformation
		      _reg_localTransformation
		      _reg_maths
		      _reg_mutualinformation
		      _reg_resampling
		      _reg_ssd
		      _reg_tools
		      reg_png
          png.lib
		      reg_nrrd
          reg_NrrdIO.lib
		      _reg_ReadWriteImage
)
