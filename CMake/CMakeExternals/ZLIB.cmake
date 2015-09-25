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

#------------------------------------------------------------------
# ZLIB
#------------------------------------------------------------------
if(MITK_USE_ZLIB)
  if(NOT DEFINED ZLIB_DIR)

    set(version "66a75305")
    set(location "${NIFTK_EP_TARBALL_LOCATION}/zlib-${version}.tar.gz")
    niftkMacroDefineExternalProjectVariables(ZLIB ${version} ${location})

    set(additional_cmake_args )
    if(CTEST_USE_LAUNCHERS)
      list(APPEND additional_cmake_args
        "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
      )
    endif()

    # Using the ZLIB from CTK:
    # https://github.com/commontk/zlib
    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${additional_cmake_args}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DZLIB_MANGLE_PREFIX:STRING=mitk_zlib_
        -DZLIB_INSTALL_INCLUDE_DIR:STRING=include/mitk_zlib
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
      )

    set(ZLIB_DIR ${proj_INSTALL})
    set(ZLIB_INCLUDE_DIR ${ZLIB_DIR}/include/mitk_zlib)

    install(DIRECTORY ${ZLIB_INCLUDE_DIR}
            DESTINATION include
            COMPONENT dev)

    find_library(ZLIB_LIBRARY_RELEASE NAMES zlib
                 PATHS ${ZLIB_DIR}
                 PATH_SUFFIXES lib lib/Release
                 NO_DEFAULT_PATH)
    find_library(ZLIB_LIBRARY_DEBUG NAMES zlibd
                 PATHS ${ZLIB_DIR}
                 PATH_SUFFIXES lib lib/Debug
                 NO_DEFAULT_PATH)

    set(ZLIB_LIBRARY )
    if(ZLIB_LIBRARY_RELEASE)
      list(APPEND ZLIB_LIBRARY ${ZLIB_LIBRARY_RELEASE})
      install(FILES ${ZLIB_LIBRARY_RELEASE}
              DESTINATION lib
              CONFIGURATIONS Release
              COMPONENT dev)
    endif()
    if(ZLIB_LIBRARY_DEBUG)
      list(APPEND ZLIB_LIBRARY ${ZLIB_LIBRARY_DEBUG})
      install(FILES ${ZLIB_LIBRARY_DEBUG}
              DESTINATION lib
              CONFIGURATIONS Debug
              COMPONENT dev)
    endif()

    mitkFunctionInstallExternalCMakeProject(${proj})
    message("SuperBuild loading ZLIB from ${ZLIB_DIR}")
    mark_as_advanced(ZLIB_LIBRARY_DEBUG)
    mark_as_advanced(ZLIB_LIBRARY_RELEASE)
    mark_as_advanced(ZLIB_LIBRARY)
  else()
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  endif()
endif()

