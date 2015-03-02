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
# ITK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED ITK_DIR AND NOT EXISTS ${ITK_DIR})
  message(FATAL_ERROR "ITK_DIR variable is defined but corresponds to non-existing directory \"${ITK_DIR}\".")
endif()

set(version "4.5.1-3e550bf8")
set(location "${NIFTK_EP_TARBALL_LOCATION}/InsightToolkit-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(ITK ${version} ${location})
set(proj_DEPENDENCIES GDCM VTK)

if(MITK_USE_Python)
  list(APPEND proj_DEPENDENCIES CableSwig)
endif()
if(BUILD_IGI)
  list(APPEND proj_DEPENDENCIES OpenCV)
endif()

if(NOT DEFINED ITK_DIR)

  set(additional_cmake_args )
  if(MINGW)
    set(additional_cmake_args
        -DCMAKE_USE_WIN32_THREADS:BOOL=ON
        -DCMAKE_USE_PTHREADS:BOOL=OFF)
  endif()

  if(MITK_USE_Python)

    list(APPEND additional_cmake_args
         -DITK_WRAPPING:BOOL=ON
         -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
         -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
         -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
         #-DPYTHON_LIBRARIES=${PYTHON_LIBRARY}
         #-DPYTHON_DEBUG_LIBRARIES=${PYTHON_DEBUG_LIBRARIES}
         -DCableSwig_DIR:PATH=${CableSwig_DIR}
         #-DITK_WRAP_JAVA:BOOL=OFF
         -DITK_WRAP_unsigned_char:BOOL=ON
         #-DITK_WRAP_double:BOOL=ON
         -DITK_WRAP_rgb_unsigned_char:BOOL=ON
         #-DITK_WRAP_rgba_unsigned_char:BOOL=ON
         -DITK_WRAP_signed_char:BOOL=ON
         #-DWRAP_signed_long:BOOL=ON
         -DITK_WRAP_signed_short:BOOL=ON
         -DITK_WRAP_short:BOOL=ON
         -DITK_WRAP_unsigned_long:BOOL=ON
        )
  else()
    list(APPEND additional_cmake_args
         -DUSE_WRAP_ITK:BOOL=OFF
        )
  endif()

  if(BUILD_IGI)
    message("Building ITK with OpenCV_DIR: ${OpenCV_DIR}")
    list(APPEND additional_cmake_args
         -DModule_ITKVideoBridgeOpenCV:BOOL=ON
         -DOpenCV_DIR:PATH=${OpenCV_DIR}
        )
  endif()

  if (BUILD_ITKFFTW)
    if(WIN32)
      # On Windows, you have to precompile one.
      # This is currently untested.
      list(APPEND additional_cmake_args
           -DUSE_SYSTEM_FFTW:BOOL=ON
          )
    else()
      # This causes FFTW to be downloaded and compiled.
      list(APPEND additional_cmake_args
           -DUSE_FFTWF:BOOL=ON
           -DUSE_FFTWD:BOOL=ON
           -DUSE_SYSTEM_FFTW:BOOL=OFF
          )
    endif()
  endif()

  # Keep the behaviour of ITK 4.3 which by default turned on ITK Review
  # see MITK bug #17338
  list(APPEND additional_cmake_args
    -DModule_ITKReview:BOOL=ON
  )

  set(ITK_PATCH_COMMAND ${CMAKE_COMMAND} -DTEMPLATE_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/EmptyFileForPatching.dummy -P ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/PatchITK-4.5.1.cmake)

  ExternalProject_Add(${proj}
    LIST_SEPARATOR ^^
    PREFIX ${proj_CONFIG}
    SOURCE_DIR ${proj_SOURCE}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${proj_LOCATION}
    URL_MD5 ${proj_CHECKSUM}
    PATCH_COMMAND ${ITK_PATCH_COMMAND}
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
      ${additional_cmake_args}
      -DITK_USE_SYSTEM_GDCM:BOOL=ON
      -DGDCM_DIR:PATH=${GDCM_DIR}
      -DVTK_DIR:PATH=${VTK_DIR}
      -DModule_ITKVtkGlue:BOOL=ON
    DEPENDS ${proj_DEPENDENCIES}
  )

  if(EP_ALWAYS_USE_INSTALL_DIR)
    set(ITK_DIR ${proj_INSTALL})
    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
  else()
    set(ITK_DIR ${proj_BUILD})
  endif()

  message("SuperBuild loading ITK from ${ITK_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
