#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/


#-----------------------------------------------------------------------------
# NIFTYSIM
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYSIM_ROOT AND NOT EXISTS ${NIFTYSIM_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYSIM_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYSIM_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYSIM)

  SET(proj NIFTYSIM)
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYSIM_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYSIM_ROOT)

    IF (DEFINED NIFTK_LOCATION_BOOST) 
      OPTION(USE_NIFTYSIM_BOOST "Enable CPU-parallelism in NiftySim through Boost." OFF)
      MARK_AS_ADVANCED(USE_NIFTYSIM_BOOST)
    ENDIF (DEFINED NIFTK_LOCATION_BOOST) 

    IF(DEFINED VTK_DIR)
      SET(USE_VTK ON)
    ELSE(DEFINED VTK_DIR)
      SET(USE_VTK OFF)
    ENDIF(DEFINED VTK_DIR)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYSIM ${NIFTK_LOCATION_NIFTYSIM})

    IF (USE_NIFTYSIM_BOOST)
      SET(proj_DEPENDENCIES BOOST)
    ELSE ()
      SET(proj_DEPENDENCIES "")
    ENDIF (USE_NIFTYSIM_BOOST)

    IF (USE_VTK)
      LIST(APPEND proj_DEPENDENCIES VTK)
    ENDIF (USE_VTK)

    ExternalProject_Add(${proj}
      URL ${NIFTK_LOCATION_NIFTYSIM}
      URL_MD5 ${NIFTK_CHECKSUM_NIFTYSIM}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DUSE_VIZ:BOOL=${USE_VTK}
	-DBoost_NO_SYSTEM_PATHS:BOOL=TRUE
        -DBoost_INCLUDE_DIR:PATH=${BOOST_INCLUDEDIR}
        -DBoost_LIBRARY_DIRS:PATH=${BOOST_LIBRARYDIR}
	-DUSE_BOOST:BOOL=${USE_NIFTYSIM_BOOST}
	-DBoost_DIR:PATH=${BOOST_ROOT}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_NIFTYSIM}
      )

    SET(NIFTYSIM_ROOT ${proj_INSTALL})
    SET(NIFTYSIM_INCLUDE_DIR "${NIFTYSIM_ROOT}/include")
    SET(NIFTYSIM_LIBRARY_DIR "${NIFTYSIM_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYSIM from ${NIFTYSIM_ROOT}")

  ELSE(NOT DEFINED NIFTYSIM_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYSIM_ROOT)

ENDIF(BUILD_NIFTYSIM)
