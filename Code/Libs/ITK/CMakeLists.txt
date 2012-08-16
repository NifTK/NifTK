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
#  Last Changed      : $LastChangedDate: 2011-12-16 14:46:48 +0000 (Fri, 16 Dec 2011) $ 
#  Revision          : $Revision: 8051 $
#  Last modified by  : $Author: be $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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


IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    INSTALL(TARGETS niftkITK 
      RUNTIME DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )
    INSTALL(TARGETS niftkITK 
      ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )    
  ELSE(WIN32)
    INSTALL(TARGETS niftkITK 
      LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
      COMPONENT libraries 
    )
  ENDIF(WIN32)
ELSE(BUILD_SHARED_LIBS)
  INSTALL(TARGETS niftkITK 
    LIBRARY DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
    ARCHIVE DESTINATION ${NIFTK_INSTALL_LIB_DIR} 
    COMPONENT libraries 
  )
ENDIF(BUILD_SHARED_LIBS)
   


