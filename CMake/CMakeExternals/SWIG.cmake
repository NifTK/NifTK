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

#------------------------------------------------------------
# SWIG (Simple Wrapper Interface Generator)
#-----------------------------------------------------------
if(DEFINED SWIG_DIR AND NOT EXISTS ${SWIG_DIR})
  message(FATAL_ERROR "SWIG_DIR variable is defined but corresponds to non-existing directory")
endif()

set(version "3.0.2")
if (WIN32)
  set(location "${NIFTK_EP_TARBALL_LOCATION}/swigwin-${version}")
else()
  set(location "${NIFTK_EP_TARBALL_LOCATION}/swig-${version}.tar.gz")
endif()
niftkMacroDefineExternalProjectVariables(SWIG ${version} ${location})
set(proj_DEPENDENCIES PCRE)

if(NOT DEFINED SWIG_DIR)

  # We don't "install" SWIG in the common install prefix,
  # since it is only used as a tool during the MITK super-build
  # to generate the Python wrappings for some projects.

  # binary SWIG for windows
  if(WIN32)
    set(swig_source_dir ${CMAKE_CURRENT_BINARY_DIR}/swigwin-${version})

    # swig.exe available as pre-built binary on Windows:
    ExternalProject_Add(${proj}
      URL ${MITK_THIRDPARTY_DOWNLOAD_PREFIX_URL}/swigwin-${version}.zip
      URL_MD5 "3f18de4fc09ab9abb0d3be37c11fbc8f"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      )

    ExternalProject_Get_Property(${proj} source_dir)
    set(SWIG_DIR ${source_dir})
    set(SWIG_EXECUTABLE ${source_dir}/swig.exe)

  else()

    # swig uses bison find it by cmake and pass it down
    find_package(BISON)
    set(BISON_FLAGS "" CACHE STRING "Flags used by bison")
    mark_as_advanced( BISON_FLAGS)

    ExternalProject_add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CONFIGURE_COMMAND <SOURCE_DIR>/./configure
                        CC=${CMAKE_C_COMPILER}${CMAKE_C_COMPILER_ARG1}
                        LDFLAGS=${CMAKE_LINKER_FLAGS} ${CMAKE_LINKER_FLAGS_RELEASE}
                        CXX=${CMAKE_CXX_COMPILER}${CMAKE_CXX_COMPILER_ARG1}
                          --prefix=<INSTALL_DIR>
                          --with-pcre-prefix=${PCRE_DIR}
                          --without-octave
                          --with-python=${PYTHON_EXECUTABLE}
      DEPENDS ${proj_DEPENDENCIES}
      )

    ExternalProject_Get_Property(${proj} install_dir)
    set(SWIG_DIR ${proj_INSTALL}/share/swig/${version})
    set(SWIG_EXECUTABLE ${proj_INSTALL}/bin/swig)
    set(NifTK_PREFIX_PATH ${SWIG_DIR}^^${NifTK_PREFIX_PATH})

  endif()

    mitkFunctionInstallExternalCMakeProject(${proj})
    message("SuperBuild loading SWIG from ${SWIG_DIR}")

else()
  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
endif()
