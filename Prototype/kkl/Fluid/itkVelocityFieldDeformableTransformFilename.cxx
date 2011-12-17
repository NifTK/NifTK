/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-27 13:54:26 +0100 (Fri, 27 May 2011) $
 Revision          : $Revision: 6300 $
 Last modified by  : $Author: kkl $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkVelocityFieldDeformableTransformFilename.h"

boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_TempPath; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_FixedImageDeformationFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_PreviousVelocityFieldGradientFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_BestDeformationFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_VelocityFieldFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_MovingImageDeformationFieldFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_BestVelocityFieldFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_StepSizeImageFilename; 
boost::filesystem::path itk::VelocityFieldDeformableTransformFilename::m_StepSizeNormalisationFactorFilename; 
