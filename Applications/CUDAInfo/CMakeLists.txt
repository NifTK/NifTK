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

if(CUDA_FOUND AND NIFTK_USE_CUDA)

  add_executable(niftkCUDAInfo niftkCUDAInfo.cxx )

  include_directories(${CUDA_TOOLKIT_INCLUDE})

  target_link_libraries(niftkCUDAInfo
    PRIVATE
      ${CUDA_CUDA_LIBRARY}
      ${CUDA_CUDART_LIBRARY}
      niftkcommon
      niftkITK
      niftkITKIO
      ${ITK_LIBRARIES}
      ${Boost_LIBRARIES}
  )

  install(TARGETS niftkCUDAInfo RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

endif()

