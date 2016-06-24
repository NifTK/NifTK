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

#--------------------------------------------------------------------------
#  PCRE (Perl Compatible Regular Expressions)
#--------------------------------------------------------------------------
if(MITK_USE_PCRE)

  if(DEFINED PCRE_DIR AND NOT EXISTS ${PCRE_DIR})
    message(FATAL_ERROR "PCRE_DIR variable is defined but corresponds to non-existing directory")
  endif()

  set(version "8.35")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/pcre-${version}.tar.gz")
  niftkMacroDefineExternalProjectVariables(PCRE ${version} ${location})
  set(proj_DEPENDENCIES)

  if(NOT DEFINED PCRE_DIR)

    if(UNIX)
      # Some other projects (e.g. Swig) require a pcre-config script which is not
      # generated when using the CMake build system.
      set(configure_cmd
        CONFIGURE_COMMAND <SOURCE_DIR>/./configure
        CC=${CMAKE_C_COMPILER}${CMAKE_C_COMPILER_ARG1}
        CFLAGS=-fPIC
        "CXXFLAGS=-fPIC ${MITK_CXX11_FLAG} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}"
        "LDFLAGS=${CMAKE_LINKER_FLAGS} ${CMAKE_LINKER_FLAGS_RELEASE} ${_install_rpath_linkflag}"
        CXX=${CMAKE_CXX_COMPILER}${CMAKE_CXX_COMPILER_ARG1}
        --prefix=<INSTALL_DIR>
        --disable-shared
        --enable-jit
      )
    else()

      set(additional_cmake_args )
      if(CTEST_USE_LAUNCHERS)
        list(APPEND additional_cmake_args
          "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
        )
      endif()

      set(configure_cmd
        CMAKE_ARGS
          ${EP_COMMON_ARGS}
          ${additional_cmake_args}
         "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} -fPIC"
         -DBUILD_SHARED_LIBS:BOOL=OFF
         -DPCRE_BUILD_PCREGREP:BOOL=OFF
         -DPCRE_BUILD_TESTS:BOOL=OFF
         -DPCRE_SUPPORT_JIT:BOOL=ON
       CMAKE_CACHE_ARGS
         ${EP_COMMON_CACHE_ARGS}
       CMAKE_CACHE_DEFAULT_ARGS
         ${EP_COMMON_CACHE_DEFAULT_ARGS}
      )

    endif()

    ExternalProject_add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      ${configure_cmd}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(PCRE_DIR ${proj_INSTALL})
    mitkFunctionInstallExternalCMakeProject(${proj})
    message("SuperBuild loading PCRE from ${PCRE_DIR}")

  else()
    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  endif()
endif()
