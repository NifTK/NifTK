#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-12-16 09:02:17 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8038 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.adnan@cs.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  # OpenMP is disabled when the compiler is based on Clang
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

ADD_DEFINITIONS(-DUSE_NIFTYREG)
LINK_DIRECTORIES(${NIFTYREG_LIBRARY_DIR})
INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})

SET(NIFTK_TEST_EXT_VTK_LINK_LIBRARIES 
  ITKIO
  vtkIO
  vtkImaging
  niftkVTK
  ${Boost_LIBRARIES}
  ${NIFTYREG_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
SET(VTK_INTEGRATION_TESTS ${CXX_TEST_PATH}/VTKIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# Build instructions.
#################################################################################
SET( VTKIntegrationTests_SRCS
)

ADD_EXECUTABLE(VTKIntegrationTests VTKIntegrationTests.cxx ${VTKIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(VTKIntegrationTests ${NIFTK_TEST_EXT_VTK_LINK_LIBRARIES} )
