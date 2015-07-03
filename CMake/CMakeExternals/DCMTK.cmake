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

# This flag should always be on. The if() statement is left so that it is easier
# to see what has changed in this file in MITK.
set(MITK_USE_DCMTK 1)

#-----------------------------------------------------------------------------
# DCMTK
#-----------------------------------------------------------------------------
if(MITK_USE_DCMTK)

  # Sanity checks
  if(DEFINED DCMTK_DIR AND NOT EXISTS ${DCMTK_DIR})
    message(FATAL_ERROR "DCMTK_DIR variable is defined but corresponds to non-existing directory")
  endif()

  set(version "3.6.1_20121102")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/dcmtk-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(DCMTK ${version} ${location})

  if(NOT DEFINED DCMTK_DIR)
    if(DCMTK_DICOM_ROOT_ID)
      set(DCMTK_CXX_FLAGS "${DCMTK_CXX_FLAGS} -DSITE_UID_ROOT=\\\"${DCMTK_DICOM_ROOT_ID}\\\"")
      set(DCMTK_C_FLAGS "${DCMTK_CXX_FLAGS} -DSITE_UID_ROOT=\\\"${DCMTK_DICOM_ROOT_ID}\\\"")
    endif()

    set(additional_args )
    if(CTEST_USE_LAUNCHERS)
      list(APPEND additional_args
        "-DCMAKE_PROJECT_${proj}_INCLUDE:FILEPATH=${CMAKE_ROOT}/Modules/CTestUseLaunchers.cmake"
      )
    endif()

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      # See http://bugs.mitk.org/show_bug.cgi?id=14513 except for the changes
      # in dcmtkMacros.cmake which allow installing release and debug executables
      # of dcmtk in the same install prefix.
      # The other patches were originally for the Xcode generator, but we always
      # apply them for consistency.
      PATCH_COMMAND ${PATCH_COMMAND} -N -p1 -i ${CMAKE_CURRENT_LIST_DIR}/DCMTK-3.6.1.patch
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${additional_args}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        #-DDCMTK_OVERWRITE_WIN32_COMPILER_FLAGS:BOOL=OFF
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} ${DCMTK_CXX_FLAGS}"
        "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} ${DCMTK_C_FLAGS}"
        #-DDCMTK_INSTALL_BINDIR:STRING=bin/${CMAKE_CFG_INTDIR}
        #-DDCMTK_INSTALL_LIBDIR:STRING=lib/${CMAKE_CFG_INTDIR}
        -DDCMTK_WITH_DOXYGEN:BOOL=OFF
        -DDCMTK_WITH_ZLIB:BOOL=OFF # see MITK bug #9894
        -DDCMTK_WITH_OPENSSL:BOOL=OFF # see MITK bug #9894
        -DDCMTK_WITH_PNG:BOOL=OFF # see MITK bug #9894
        -DDCMTK_WITH_TIFF:BOOL=OFF  # see MITK bug #9894
        -DDCMTK_WITH_XML:BOOL=OFF  # see MITK bug #9894
        -DDCMTK_WITH_ICONV:BOOL=OFF  # see MITK bug #9894
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(DCMTK_DIR ${proj_INSTALL})

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading DCMTK from ${DCMTK_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()
endif()
