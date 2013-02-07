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

SET(niftkITK_SRCS
Common/itkConversionUtils.cxx
Common/itkCommandLineHelper.cxx
Common/itkFilenameUtils.cxx
Common/itkLogHelper.cxx
RegistrationToolbox/Optimizers/itkUCLSimplexOptimizer.cxx
RegistrationToolbox/Optimizers/itkUCLRegularStepGradientDescentOptimizer.cxx
RegistrationToolbox/Optimizers/itkUCLRegularStepOptimizer.cxx
RegistrationToolbox/Optimizers/itkUCLPowellOptimizer.cxx
RegistrationToolbox/Commands/itkIterationUpdateCommand.cxx
RegistrationToolbox/Commands/itkVnlIterationUpdateCommand.cxx 
IO/itkINRImageIO.cxx
IO/itkINRImageIOFactory.cxx
IO/itkDRCAnalyzeImageIO.cxx
IO/itkDRCAnalyzeImageIOFactory.cxx
IO/itkNIFTKTransformIOFactory.cxx
IO/itkNIFTKTransformIO.cxx
2D3DToolbox/Commands/itkReconstructionUpdateCommand.cxx
2D3DToolbox/Commands/itkReconstructionAndRegistrationUpdateCommand.cxx
2D3DToolbox/Commands/itkSimultaneousReconAndRegnUpdateCommand.cxx
Segmentation/itkMIDASHelper.cxx
)

ADD_LIBRARY(niftkITK ${niftkITK_SRCS})

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKITK_WINDOWS_EXPORT)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)

TARGET_LINK_LIBRARIES(niftkITK  
  niftkcommon
  ${NIFTK_ITK_LIB}
  ${Boost_LIBRARIES}  
)

MITK_INSTALL(TARGETS niftkITK)

