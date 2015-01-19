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
# VTK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED VTK_DIR AND NOT EXISTS ${VTK_DIR})
  message(FATAL_ERROR "VTK_DIR variable is defined but corresponds to non-existing directory \"${VTK_DIR}\".")
endif()

niftkMacroGetCommitHashOfCurrentFile(config_version)

set(proj VTK)
set(proj_VERSION ${NIFTK_VERSION_${proj}})
set(proj_SOURCE ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-src)
set(proj_CONFIG ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-cmake)
set(proj_BUILD ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-build)
set(proj_INSTALL ${EP_BASE}/${proj}-${proj_VERSION}-${config_version}-install)
set(proj_DEPENDENCIES )
set(VTK_DEPENDS ${proj})

set(VTK_PATCH_COMMAND ${CMAKE_COMMAND} -DTEMPLATE_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/EmptyFileForPatching.dummy -P ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/PatchVTK.cmake)

if(NOT DEFINED VTK_DIR)

  if(WIN32)
    option(VTK_USE_SYSTEM_FREETYPE OFF)
  else(WIN32)
    option(VTK_USE_SYSTEM_FREETYPE ON)
  endif(WIN32)

  set(additional_cmake_args )
  if(MINGW)
    set(additional_cmake_args
        -DCMAKE_USE_WIN32_THREADS:BOOL=ON
        -DCMAKE_USE_PTHREADS:BOOL=OFF
        -DVTK_USE_VIDEO4WINDOWS:BOOL=OFF # no header files provided by MinGW
        )
  endif(MINGW)

  if(WIN32)
    # see http://bugs.mitk.org/show_bug.cgi?id=17858
    list(APPEND additional_cmake_args
         -DVTK_DO_NOT_DEFINE_OSTREAM_SLL:BOOL=ON
         -DVTK_DO_NOT_DEFINE_OSTREAM_ULL:BOOL=ON
        )
  endif()

  # Optionally enable memory leak checks for any objects derived from vtkObject. This
  # will force unit tests to fail if they have any of these memory leaks.
  option(MITK_VTK_DEBUG_LEAKS OFF)
  mark_as_advanced(MITK_VTK_DEBUG_LEAKS)
  list(APPEND additional_cmake_args
       -DVTK_DEBUG_LEAKS:BOOL=${MITK_VTK_DEBUG_LEAKS}
      )


  list(APPEND additional_cmake_args
      -DVTK_QT_VERSION:STRING=${DESIRED_QT_VERSION}
      -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
      -DModule_vtkGUISupportQt:BOOL=ON
      -DModule_vtkGUISupportQtWebkit:BOOL=ON
      -DModule_vtkGUISupportQtSQL:BOOL=ON
      -DModule_vtkRenderingQt:BOOL=ON
      -DVTK_Group_Qt:BOOL=ON
  )

  if(APPLE)
    set(additional_cmake_args
        ${additional_cmake_args}
        -DVTK_REQUIRED_OBJCXX_FLAGS:STRING=""
        )
  endif(APPLE)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_VTK ${NIFTK_LOCATION_VTK})

  ExternalProject_Add(${proj}
    SOURCE_DIR ${proj_SOURCE}
    PREFIX ${proj_CONFIG}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${NIFTK_LOCATION_VTK}
    URL_MD5 ${NIFTK_CHECKSUM_VTK}
    PATCH_COMMAND ${VTK_PATCH_COMMAND}
    INSTALL_COMMAND ""
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DVTK_WRAP_TCL:BOOL=OFF
        -DVTK_WRAP_PYTHON:BOOL=OFF
        -DVTK_WRAP_JAVA:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DVTK_USE_RPATH:BOOL=ON
        -DVTK_USE_SYSTEM_FREETYPE:BOOL=${VTK_USE_SYSTEM_FREETYPE}
        -DVTK_USE_GUISUPPORT:BOOL=ON
        -DVTK_LEGACY_REMOVE:BOOL=ON
        -DModule_vtkTestingRendering:BOOL=ON
        -DVTK_MAKE_INSTANTIATORS:BOOL=ON
        ${additional_cmake_args}
        ${VTK_QT_ARGS}
    DEPENDS ${proj_DEPENDENCIES}
    )

  set(VTK_DIR ${proj_BUILD})
  message("SuperBuild loading VTK from ${VTK_DIR}")

else(NOT DEFINED VTK_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED VTK_DIR)
