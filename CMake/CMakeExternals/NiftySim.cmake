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
# NiftySim
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYSIM_ROOT AND NOT EXISTS ${NIFTYSIM_ROOT})
  message(FATAL_ERROR "NIFTYSIM_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYSIM_ROOT}\".")
endif()

if(BUILD_NIFTYSIM)
  set(proj NiftySim)
  set(proj_INSTALL ${CMAKE_BINARY_DIR}/${proj}-install )
  set(NIFTYSIM_DEPENDS ${proj})

  if(NOT DEFINED NIFTYSIM_ROOT)
    if (DEFINED NIFTK_LOCATION_Boost)
      option(USE_NIFTYSIM_BOOST "Enable CPU-parallelism in NiftySim through Boost." OFF)
      mark_as_advanced(USE_NIFTYSIM_BOOST)
    endif (DEFINED NIFTK_LOCATION_Boost)

    if(DEFINED VTK_DIR)
      set(USE_VTK ON)
    else(DEFINED VTK_DIR)
      set(USE_VTK OFF)
    endif(DEFINED VTK_DIR)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYSIM ${NIFTK_LOCATION_NIFTYSIM})

    set(proj_DEPENDENCIES "")
    if (USE_NIFTYSIM_BOOST)
      list(APPEND proj_DEPENDENCIES Boost)
    endif (USE_NIFTYSIM_BOOST)

    if (USE_VTK)
      list(APPEND proj_DEPENDENCIES VTK)
    endif (USE_VTK)

    # Temporary solution to build problems on MSVS 2010
    if (WIN32)
      set(NIFTYSIM_CMAKE_CXX_FLAGS "/D_USE_MATH_DEFINES")
    else ()
      set(NIFTYSIM_CMAKE_CXX_FLAGS "")
    endif (WIN32)

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_NIFTYSIM}
      URL_MD5 ${NIFTK_CHECKSUM_NIFTYSIM}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DUSE_BOOST:BOOL=${USE_NIFTYSIM_BOOST}
        -DUSE_VIZ:BOOL=${USE_VTK}
        -DVTK_DIR:PATH=${VTK_DIR}
        -DVTK_INCLUDE_DIRS=${VTK_INCLUDE_DIRS}
        -DVTK_LIBRARY_DIRS=${VTK_LIBRARY_DIRS}
        -DBoost_NO_SYSTEM_PATHS:BOOL=TRUE
        -DBOOST_ROOT:PATH=${BOOST_ROOT}
        -DBoost_USE_STATIC_LIBS:BOOL=TRUE
        -DBOOST_INCLUDEDIR:PATH=${BOOST_INCLUDEDIR}
        -DBOOST_LIBRARYDIR:PATH=${BOOST_LIBRARYDIR}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DCMAKE_CXX_FLAGS:STRING=${NIFTYSIM_CMAKE_CXX_FLAGS}
      DEPENDS ${proj_DEPENDENCIES}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_NIFTYSIM}
      )

    set(NIFTYSIM_ROOT ${proj_INSTALL})
    set(NIFTYSIM_INCLUDE_DIR "${NIFTYSIM_ROOT}/include")
    set(NIFTYSIM_LIBRARY_DIR "${NIFTYSIM_ROOT}/lib")

    message("SuperBuild loading NiftySim from ${NIFTYSIM_ROOT}")

  else(NOT DEFINED NIFTYSIM_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYSIM_ROOT)

endif(BUILD_NIFTYSIM)
