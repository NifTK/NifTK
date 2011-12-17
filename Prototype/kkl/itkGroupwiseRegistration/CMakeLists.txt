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
#  Last Changed      : $LastChangedDate: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $ 
#  Revision          : $Revision: 7333 $
#  Last modified by  : $Author: ad $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
ADD_EXECUTABLE(itkGroupwiseRegistrationTest itkGroupwiseRegistrationTest.cxx)
ADD_EXECUTABLE(itkMultipleImageRegistrationTest itkMulitpleImageRegistrationTest.cxx)
ADD_EXECUTABLE(itkTransformTest itkTransformTest.cxx)
ADD_EXECUTABLE(itkWatershedSegmentationTest itkWatershedSegmentationTest.cxx)
ADD_EXECUTABLE(itkDistanceMapTest itkDistanceMapTest.cxx)
ADD_EXECUTABLE(itkDeformableModelTest itkDeformableModelTest.cxx)
ADD_EXECUTABLE(itkKmeansClassifierTest itkKmeansClassifierTest.cxx)
ADD_EXECUTABLE(itkConnectedComponentTest itkConnectedComponentTest.cxx)
ADD_EXECUTABLE(itkLevelSetTest itkLevelSetTest.cxx)
ADD_EXECUTABLE(itkResampleTest itkResampleTest.cxx)
ADD_EXECUTABLE(itkAddImageTest itkAddImageTest.cxx)
ADD_EXECUTABLE(itkSimpleAverage itkSimpleAverage.cxx)
ADD_EXECUTABLE(itkComputeNewGroupwiseTransformations itkComputeNewGroupwiseTransformations.cxx)
ADD_EXECUTABLE(itkComputeInitialAtlas itkComputeInitialAtlas.cxx)
ADD_EXECUTABLE(itkComputeInitialAffineAtlas itkComputeInitialAffineAtlas.cxx)
ADD_EXECUTABLE(itkComputeImageDifference itkComputeImageDifference.cxx)
ADD_EXECUTABLE(itkInvertTransformation itkInvertTransformation.cxx)
ADD_EXECUTABLE(niftkMultiplyTransformation niftkMultiplyTransformation.cxx)
ADD_EXECUTABLE(itkComputeMeanTransformation itkComputeMeanTransformation.cxx)
ADD_EXECUTABLE(ComputeSymmetricAtrophyRate ComputeSymmetricAtrophyRate.cxx)
ADD_EXECUTABLE(niftkMedian niftkMedian.cxx)
ADD_EXECUTABLE(niftkKBilateralImageFilter niftkKBilateralImageFilter.cxx)
ADD_EXECUTABLE(niftkCropMidasStr niftkCropMidasStr.cxx)    
ADD_EXECUTABLE(niftkConvertNiiToMidasStr niftkConvertNiiToMidasStr.cxx)
ADD_EXECUTABLE(niftkSetImageDirection niftkSetImageDirection.cxx)


TARGET_LINK_LIBRARIES(itkGroupwiseRegistrationTest  
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(itkMultipleImageRegistrationTest  
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES}  
  )

TARGET_LINK_LIBRARIES(itkTransformTest  
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES}
  )
  
TARGET_LINK_LIBRARIES(itkWatershedSegmentationTest  
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO    
  ${Boost_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(itkDistanceMapTest 
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO
  ${Boost_LIBRARIES} 
  )

TARGET_LINK_LIBRARIES(itkDeformableModelTest
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES}   
  )
  
TARGET_LINK_LIBRARIES(itkKmeansClassifierTest
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )


TARGET_LINK_LIBRARIES(itkConnectedComponentTest
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(itkLevelSetTest
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(itkResampleTest
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES}   
  )

TARGET_LINK_LIBRARIES(itkAddImageTest
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES} 
  )

TARGET_LINK_LIBRARIES(itkSimpleAverage
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )
  
TARGET_LINK_LIBRARIES(itkComputeNewGroupwiseTransformations
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES} 
  )

TARGET_LINK_LIBRARIES(itkComputeInitialAtlas
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO
  ${Boost_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(itkComputeInitialAffineAtlas
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO
  ${Boost_LIBRARIES}
  )
INSTALL(TARGETS itkComputeInitialAffineAtlas RUNTIME DESTINATION bin COMPONENT applications)

  
TARGET_LINK_LIBRARIES(itkComputeImageDifference
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(itkInvertTransformation
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )

TARGET_LINK_LIBRARIES(niftkMultiplyTransformation
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )

TARGET_LINK_LIBRARIES(itkComputeMeanTransformation
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )
  
TARGET_LINK_LIBRARIES(niftkMedian
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )
  
TARGET_LINK_LIBRARIES(niftkKBilateralImageFilter
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )
   
TARGET_LINK_LIBRARIES(niftkCropMidasStr
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )
  
TARGET_LINK_LIBRARIES(niftkConvertNiiToMidasStr
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )
  
TARGET_LINK_LIBRARIES(niftkSetImageDirection
  niftkcommon   
  niftkITK 
  ITKCommon 
  ITKIO   
  ${Boost_LIBRARIES} 
  )
  
INSTALL(TARGETS itkComputeMeanTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS itkInvertTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS niftkMultiplyTransformation RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS ComputeSymmetricAtrophyRate RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS niftkMedian RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS niftkKBilateralImageFilter RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS itkResampleTest RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS niftkCropMidasStr RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS niftkConvertNiiToMidasStr RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)

INSTALL(TARGETS niftkSetImageDirection RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)


