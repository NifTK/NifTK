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
# DCMTK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED DCMTK_DIR AND NOT EXISTS ${DCMTK_DIR})
  message(FATAL_ERROR "DCMTK_DIR variable is defined but corresponds to non-existing directory")
endif()

set(proj DCMTK)
set(proj_DEPENDENCIES )
set(DCMTK_DEPENDS ${proj})

if(NOT DEFINED DCMTK_DIR)

  if(UNIX)
    set(DCMTK_CXX_FLAGS "-fPIC")
    set(DCMTK_C_FLAGS "-fPIC")
  endif(UNIX)
  if(DCMTK_DICOM_ROOT_ID)
    set(DCMTK_CXX_FLAGS "${DCMTK_CXX_FLAGS} -DSITE_UID_ROOT=\\\"${DCMTK_DICOM_ROOT_ID}\\\"")
    set(DCMTK_C_FLAGS "${DCMTK_CXX_FLAGS} -DSITE_UID_ROOT=\\\"${DCMTK_DICOM_ROOT_ID}\\\"")
  endif()

  niftkMacroGetChecksum(NIFTK_CHECKSUM_DCMTK ${NIFTK_LOCATION_DCMTK})

  ExternalProject_Add(${proj}
    URL ${NIFTK_LOCATION_DCMTK}
    URL_MD5 ${NIFTK_CHECKSUM_DCMTK}
    SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}-src
    BINARY_DIR ${proj}-build
    PREFIX ${proj}-cmake
    INSTALL_DIR ${proj}-install
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
       ${ep_common_args}
       -DDCMTK_OVERWRITE_WIN32_COMPILER_FLAGS:BOOL=OFF
       -DBUILD_SHARED_LIBS:BOOL=OFF
       "-DCMAKE_CXX_FLAGS:STRING=${ep_common_CXX_FLAGS} ${DCMTK_CXX_FLAGS}"
       "-DCMAKE_C_FLAGS:STRING=${ep_common_C_FLAGS} ${DCMTK_C_FLAGS}"
       -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/${proj}-install
       -DDCMTK_WITH_DOXYGEN:BOOL=OFF
       -DDCMTK_WITH_ZLIB:BOOL=OFF # see MITK bug #9894
       -DDCMTK_WITH_OPENSSL:BOOL=OFF # see MITK bug #9894
       -DDCMTK_WITH_PNG:BOOL=OFF # see MITK bug #9894
       -DDCMTK_WITH_TIFF:BOOL=OFF  # see MITK bug #9894
       -DDCMTK_WITH_XML:BOOL=OFF  # see MITK bug #9894
       -DDCMTK_WITH_ICONV:BOOL=OFF  # see bug #9894
       -DDCMTK_FORCE_FPIC_ON_UNIX:BOOL=ON
    DEPENDS ${proj_DEPENDENCIES}
    )
  set(DCMTK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-install)
  MESSAGE("SuperBuild loading DCMTK from ${DCMTK_DIR}")
  
else()
  
  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
endif()
