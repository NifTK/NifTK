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
#  Last Changed      : $LastChangedDate: 2010-07-06 14:07:20 +0100 (Tue, 06 Jul 2010) $ 
#  Revision          : $Revision: 3467 $
#  Last modified by  : $Author: jhh $
#
#  Original author   : j.hipwell@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

SET(NIFTK_APPLICATION_LINK_LIBRARIES 
  niftkcommon 
  niftkITK
  ITKCommon 
  ITKIO 
  ITKBasicFilters
  ITKAlgorithms
  itkzlib
  ${Boost_LIBRARIES} 
  )

# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

ADD_EXECUTABLE(niftkSimultaneousAffineRegistrationAndReconstruction niftkSimultaneousAffineRegistrationAndReconstruction.cxx )
TARGET_LINK_LIBRARIES(niftkSimultaneousAffineRegistrationAndReconstruction ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSimultaneousAffineRegistrationAndReconstruction RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkCreateForwardBackwardProjectionMatrix niftkCreateForwardBackwardProjectionMatrix.cxx )
TARGET_LINK_LIBRARIES(niftkCreateForwardBackwardProjectionMatrix ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkCreateForwardBackwardProjectionMatrix RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkCreateEulerAffineTransformMatrix niftkCreateEulerAffineTransformMatrix.cxx )
TARGET_LINK_LIBRARIES(niftkCreateEulerAffineTransformMatrix ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkCreateEulerAffineTransformMatrix RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkEulerAffineTransformMatrixAndItsVariations niftkEulerAffineTransformMatrixAndItsVariations.cxx )
TARGET_LINK_LIBRARIES(niftkEulerAffineTransformMatrixAndItsVariations ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkEulerAffineTransformMatrixAndItsVariations RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkForwardAndBackwardProjectionMatrix niftkForwardAndBackwardProjectionMatrix.cxx )
TARGET_LINK_LIBRARIES(niftkForwardAndBackwardProjectionMatrix ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkForwardAndBackwardProjectionMatrix RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkSimultaneousUnconstrainedMatrixReconRegnMetric niftkSimultaneousUnconstrainedMatrixReconRegnMetric.cxx )
TARGET_LINK_LIBRARIES(niftkSimultaneousUnconstrainedMatrixReconRegnMetric ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSimultaneousUnconstrainedMatrixReconRegnMetric RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkSimultaneousUnconstrainedAffineReconstructionAndRegistration niftkSimultaneousUnconstrainedAffineReconstructionAndRegistration.cxx )
TARGET_LINK_LIBRARIES(niftkSimultaneousUnconstrainedAffineReconstructionAndRegistration ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkSimultaneousUnconstrainedAffineReconstructionAndRegistration RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkImageMatrixFormReconstruction niftkImageMatrixFormReconstruction.cxx )
TARGET_LINK_LIBRARIES(niftkImageMatrixFormReconstruction ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkImageMatrixFormReconstruction RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkImageMatrixFormReconTwoDataSetsWithoutReg niftkImageMatrixFormReconTwoDataSetsWithoutReg.cxx )
TARGET_LINK_LIBRARIES(niftkImageMatrixFormReconTwoDataSetsWithoutReg ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkImageMatrixFormReconTwoDataSetsWithoutReg RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkMatrixBasedSimulUnconReconRegn niftkMatrixBasedSimulUnconReconRegn.cxx )
TARGET_LINK_LIBRARIES(niftkMatrixBasedSimulUnconReconRegn ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMatrixBasedSimulUnconReconRegn RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkMatrixBasedAffineParametersGradientTest niftkMatrixBasedAffineParametersGradientTest.cxx )
TARGET_LINK_LIBRARIES(niftkMatrixBasedAffineParametersGradientTest ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkMatrixBasedAffineParametersGradientTest RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkInnerProductAffineParametersGradientTest niftkInnerProductAffineParametersGradientTest.cxx )
TARGET_LINK_LIBRARIES(niftkInnerProductAffineParametersGradientTest ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkInnerProductAffineParametersGradientTest RUNTIME DESTINATION bin COMPONENT applications)

ADD_EXECUTABLE(niftkParameterGradientTest niftkParameterGradientTest.cxx )
TARGET_LINK_LIBRARIES(niftkParameterGradientTest ${NIFTK_APPLICATION_LINK_LIBRARIES} )
INSTALL(TARGETS niftkParameterGradientTest RUNTIME DESTINATION bin COMPONENT applications)
