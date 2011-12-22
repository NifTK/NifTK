#-----------------------------------------------------------------------------
# NIFTYSEG
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYSEG_ROOT AND NOT EXISTS ${NIFTYSEG_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYSEG_ROOT variable is defined but corresponds to non-existing disegtory \"${NIFTYSEG_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYSEG)

  SET(proj NIFTYSEG)
  SET(proj_DEPENDENCIES )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYSEG_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYSEG_ROOT)

    ExternalProject_Add(${proj}
      SVN_REPOSITORY https://niftyseg.svn.sourceforge.net/svnroot/niftyseg
      SVN_REVISION -r 28
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED}
	      -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
	      -DINSTALL_PRIORS:BOOL=ON
	      -DINSTALL_PRIORS_DIRECTORY:PATH=${EP_BASE}/Install/${proj}/priors
	      -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )

    SET(NIFTYSEG_ROOT ${proj_INSTALL})
    SET(NIFTYSEG_INCLUDE_DIR "${NIFTYSEG_ROOT}/include")
    SET(NIFTYSEG_LIBRARY_DIR "${NIFTYSEG_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYSEG from ${NIFTYSEG_ROOT}")

  ELSE(NOT DEFINED NIFTYSEG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYSEG_ROOT)

ENDIF(BUILD_NIFTYSEG)
