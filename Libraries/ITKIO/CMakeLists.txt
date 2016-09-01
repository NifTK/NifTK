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

set(niftkITKIO_SRCS
  itkINRImageIO.cxx
  itkAnalyzeDbh.cxx
  itkAnalyzeImageIO3160.cxx
  itkDRCAnalyzeImageIO.cxx
  itkNifTKImageIOFactory.cxx
  itkNiftiImageIO3201.cxx
)

add_library(niftkITKIO ${niftkITKIO_SRCS})

if(BUILD_SHARED_LIBS)
  if(WIN32)
    add_definitions(-DNIFTKITKIO_WINDOWS_EXPORT=1)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)

target_include_directories(niftkITKIO
  PUBLIC
    ${ITK_INCLUDE_DIRS}
)

target_link_libraries(niftkITKIO
  PUBLIC
    niftkcommon
    ${ITK_LIBRARIES}
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkITKIO)
endif()

MITK_INSTALL(TARGETS niftkITKIO)
