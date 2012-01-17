#-----------------------------------------------------------------------------
# NIFTYREG
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYREG_ROOT AND NOT EXISTS ${NIFTYREG_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYREG_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREG_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYREG)

  SET(proj NIFTYREG)
  SET(proj_DEPENDENCIES )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYREG_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYREG_ROOT)

    ExternalProject_Add(${proj}
      SVN_REPOSITORY https://niftyreg.svn.sourceforge.net/svnroot/niftyreg/trunk/nifty_reg/
      SVN_REVISION -r 239
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
	      -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
	      -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )

    SET(NIFTYREG_ROOT ${proj_INSTALL})
    SET(NIFTYREG_INCLUDE_DIR "${NIFTYREG_ROOT}/include")
    SET(NIFTYREG_LIBRARY_DIR "${NIFTYREG_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYREG from ${NIFTYREG_ROOT}")

  ELSE(NOT DEFINED NIFTYREG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYREG_ROOT)

ENDIF(BUILD_NIFTYREG)
