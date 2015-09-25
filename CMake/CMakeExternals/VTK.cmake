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

set(version "6.2.0")
set(location "${NIFTK_EP_TARBALL_LOCATION}/VTK-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(VTK ${version} ${location})

if(MITK_USE_HDF5)
  list(APPEND proj_DEPENDENCIES HDF5)
endif()

if(NOT DEFINED VTK_DIR)

  if(WIN32)
    option(VTK_USE_SYSTEM_FREETYPE OFF)
  else(WIN32)
    option(VTK_USE_SYSTEM_FREETYPE ON)
  endif(WIN32)
  mark_as_advanced(VTK_USE_SYSTEM_FREETYPE)

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

  if(MITK_USE_Python)
    if(NOT MITK_USE_SYSTEM_PYTHON)
     list(APPEND proj_DEPENDENCIES Python)
     set(_vtk_install_python_dir -DVTK_INSTALL_PYTHON_MODULE_DIR:FILEPATH=${MITK_PYTHON_SITE_DIR})
    else()
      set(_vtk_install_python_dir -DVTK_INSTALL_PYTHON_MODULE_DIR:PATH=${EP_BASE}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
    endif()

    list(APPEND additional_cmake_args
         -DVTK_WRAP_PYTHON:BOOL=ON
         -DVTK_USE_TK:BOOL=OFF
         -DVTK_WINDOWS_PYTHON_DEBUGGABLE:BOOL=OFF
         -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
         -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
         -DPYTHON_INCLUDE_DIR2:PATH=${PYTHON_INCLUDE_DIR2}
         -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
         ${_vtk_install_python_dir}
        )
  else()
    list(APPEND additional_cmake_args
         -DVTK_WRAP_PYTHON:BOOL=OFF
         -DVTK_WINDOWS_PYTHON_DEBUGGABLE:BOOL=OFF
        )
  endif()

  if(MITK_USE_QT)
    list(APPEND additional_cmake_args
        -DVTK_QT_VERSION:STRING=${DESIRED_QT_VERSION}
        -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE}
        -DModule_vtkGUISupportQt:BOOL=ON
        -DModule_vtkGUISupportQtWebkit:BOOL=ON
        -DModule_vtkGUISupportQtSQL:BOOL=ON
        -DModule_vtkRenderingQt:BOOL=ON
        -DVTK_Group_Qt:BOOL=ON
    )
  endif()

  if(APPLE)
    set(additional_cmake_args
        ${additional_cmake_args}
        -DVTK_REQUIRED_OBJCXX_FLAGS:STRING=""
        )
  endif(APPLE)

  if(CTEST_USE_LAUNCHERS)
    list(APPEND additional_cmake_args
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
    PATCH_COMMAND ${PATCH_COMMAND} -N -p1 -i ${CMAKE_CURRENT_LIST_DIR}/VTK-6.2.0.patch
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DVTK_WRAP_TCL:BOOL=OFF
        -DVTK_WRAP_PYTHON:BOOL=OFF
        -DVTK_WRAP_JAVA:BOOL=OFF
        -DVTK_USE_RPATH:BOOL=ON
        -DVTK_USE_SYSTEM_FREETYPE:BOOL=${VTK_USE_SYSTEM_FREETYPE}
        -DVTK_USE_GUISUPPORT:BOOL=ON
        -DVTK_LEGACY_REMOVE:BOOL=ON
        -DModule_vtkTestingRendering:BOOL=ON
        -DVTK_MAKE_INSTANTIATORS:BOOL=ON
        -DVTK_REPORT_OPENGL_ERRORS:BOOL=OFF
        ${additional_cmake_args}
    CMAKE_CACHE_ARGS
      ${EP_COMMON_CACHE_ARGS}
    CMAKE_CACHE_DEFAULT_ARGS
      ${EP_COMMON_CACHE_DEFAULT_ARGS}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(VTK_DIR ${proj_INSTALL})
  set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
  mitkFunctionInstallExternalCMakeProject(${proj})

  message("SuperBuild loading VTK from ${VTK_DIR}")
  if(MITK_USE_Python)
    message("SuperBuild loading VTK Python from ${_vtk_install_python_dir}")
  endif()

else(NOT DEFINED VTK_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED VTK_DIR)
