#-----------------------------------------------------------------------------
# NIFTYREC
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYREC_ROOT AND NOT EXISTS ${NIFTYREC_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYREC_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYREC_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYREC)

  SET(proj NIFTYREC)
  SET(proj_DEPENDENCIES NIFTYREG )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYREC_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYREC_ROOT)

    ExternalProject_Add(${proj}
      SVN_REPOSITORY https://niftyrec.svn.sourceforge.net/svnroot/niftyrec/
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DCUDA_SDK_ROOT_DIR=${CUDA_SDK_ROOT_DIR}
      DEPENDS ${proj_DEPENDENCIES}
      )

    SET(NIFTYREC_ROOT ${proj_INSTALL})
    SET(NIFTYREC_INCLUDE_DIR "${NIFTYREC_ROOT}/include")
    SET(NIFTYREC_LIBRARY_DIR "${NIFTYREC_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYREC from ${NIFTYREC_ROOT}")

  ELSE(NOT DEFINED NIFTYREC_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYREC_ROOT)

ENDIF(BUILD_NIFTYREC)
