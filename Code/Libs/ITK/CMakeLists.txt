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

set(niftkITK_SRCS
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
  IO/itkAnalyzeImageIO.cxx
  IO/itkAnalyzeImageIOFactory.cxx
  IO/itkDRCAnalyzeImageIO.cxx
  IO/itkDRCAnalyzeImageIOFactory.cxx
  IO/itkNIFTKTransformIOFactory.cxx
  IO/itkNIFTKTransformIO.cxx
  2D3DToolbox/Commands/itkReconstructionUpdateCommand.cxx
  2D3DToolbox/Commands/itkReconstructionAndRegistrationUpdateCommand.cxx
  2D3DToolbox/Commands/itkSimultaneousReconAndRegnUpdateCommand.cxx
  Segmentation/itkMIDASHelper.cxx
)

add_library(niftkITK ${niftkITK_SRCS})

if(BUILD_SHARED_LIBS)
  if(WIN32)
    add_definitions(-DNIFTKITK_WINDOWS_EXPORT)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)

target_link_libraries(niftkITK
  niftkcommon
  ${ITK_LIBRARIES}
  ${Boost_LIBRARIES}
)

MITK_INSTALL(TARGETS niftkITK)
