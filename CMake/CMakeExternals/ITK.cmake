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

set(proj ITK)
set(proj_DEPENDENCIES GDCM)
if(MITK_USE_Python)
  list(APPEND proj_DEPENDENCIES CableSwig)
endif()
if(BUILD_IGI)
  list(APPEND proj_DEPENDENCIES OpenCV)
endif()

set(ITK_DEPENDS ${proj})

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
         #-DITK_USE_REVIEW:BOOL=ON
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
  
  set(ITK_PATCH_COMMAND ${CMAKE_COMMAND} -DTEMPLATE_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/EmptyFileForPatching.dummy -P ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/PatchITK-4.3.1.cmake)

  niftkMacroGetChecksum(NIFTK_CHECKSUM_ITK ${NIFTK_LOCATION_ITK})

  ExternalProject_Add(${proj}
    SOURCE_DIR ${proj}-src
    BINARY_DIR ${proj}-build
    PREFIX ${proj}-cmake
    INSTALL_DIR ${proj}-install
    URL ${NIFTK_LOCATION_ITK}
    URL_MD5 ${NIFTK_CHECKSUM_ITK}
    INSTALL_COMMAND ""
    PATCH_COMMAND ${ITK_PATCH_COMMAND}
    CMAKE_GENERATOR ${GEN}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      ${additional_cmake_args}
      -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
      -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
      -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
      -DITK_USE_SYSTEM_GDCM:BOOL=ON
      -DGDCM_DIR:PATH=${GDCM_DIR}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(ITK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  message("SuperBuild loading ITK from ${ITK_DIR}")

else()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif()
