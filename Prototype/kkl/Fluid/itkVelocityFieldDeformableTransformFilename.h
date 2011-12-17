/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-26 10:49:56 +0100 (Thu, 26 May 2011) $
 Revision          : $Revision: 6271 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKVelocityFieldDeformableTransformFilename_H_
#define ITKVelocityFieldDeformableTransformFilename_H_

#include "boost/filesystem.hpp"
#include "sstream"

namespace itk
{
  
/** 
 * \class VelocityFieldDeformableTransformFilename
 * \brief Filenames for various temp files, e.g. velocity fields.
 */
class VelocityFieldDeformableTransformFilename
{
public:
  static const char* GetFixedImageDeformationFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "fixed_image_deformation_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_FixedImageDeformationFilename = m_TempPath; 
    m_FixedImageDeformationFilename /= filenamePath; 
    return m_FixedImageDeformationFilename.string().c_str(); 
  }
  
  static const char* GetPreviousVelocityFieldGradientFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "previous_velocity_field_gradient_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_PreviousVelocityFieldGradientFilename = m_TempPath; 
    m_PreviousVelocityFieldGradientFilename /= filenamePath; 
    return m_PreviousVelocityFieldGradientFilename.string().c_str(); 
  }
  
  static const char* GetBestDeformationFilename()
  {
    std::ostringstream filenameString; 
    filenameString << "best_deformation_20.nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_PreviousVelocityFieldGradientFilename = m_TempPath; 
    m_PreviousVelocityFieldGradientFilename /= filenamePath; 
    return m_PreviousVelocityFieldGradientFilename.string().c_str(); 
  }
  
  static const char* GetVelocityFieldFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "velocity_field_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_VelocityFieldFilename = m_TempPath; 
    m_VelocityFieldFilename /= filenamePath; 
    return m_VelocityFieldFilename.string().c_str(); 
  }
  
  static const char* GetMovingImageDeformationFieldFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "moving_image_deformation_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_MovingImageDeformationFieldFilename = m_TempPath; 
    m_MovingImageDeformationFieldFilename /= filenamePath; 
    return m_MovingImageDeformationFieldFilename.string().c_str(); 
  }
  
  static const char* GetBestVelocityFieldFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "best_velocity_field_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_BestVelocityFieldFilename = m_TempPath; 
    m_BestVelocityFieldFilename /= filenamePath; 
    return m_BestVelocityFieldFilename.string().c_str(); 
  }
  
  static const char* GetStepSizeImageFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "step_size_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_StepSizeImageFilename = m_TempPath; 
    m_StepSizeImageFilename /= filenamePath; 
    return m_StepSizeImageFilename.string().c_str(); 
  }
  
  static const char* GetStepSizeNormalisationFactorFilename(int timePoint)
  {
    std::ostringstream filenameString; 
    filenameString << "step_size_normalisation_factor_" << timePoint << ".nii"; 
    boost::filesystem::path filenamePath(filenameString.str()); 
    
    m_StepSizeNormalisationFactorFilename = m_TempPath; 
    m_StepSizeNormalisationFactorFilename /= filenamePath; 
    return m_StepSizeNormalisationFactorFilename.string().c_str(); 
  }
  
  static void SetTempPath(const char* tempPath)
  {
    m_TempPath = tempPath; 
  }

  static const char* GetTempPath()
  {
    return m_TempPath.string().c_str(); 
  }
  
protected:
  static boost::filesystem::path m_TempPath; 
  
  static boost::filesystem::path m_FixedImageDeformationFilename; 
  
  static boost::filesystem::path m_PreviousVelocityFieldGradientFilename; 
  
  static boost::filesystem::path m_BestDeformationFilename; 
  
  static boost::filesystem::path m_VelocityFieldFilename; 
  
  static boost::filesystem::path m_MovingImageDeformationFieldFilename; 
  
  static boost::filesystem::path m_BestVelocityFieldFilename; 
  
  static boost::filesystem::path m_StepSizeImageFilename; 
  
  static boost::filesystem::path m_StepSizeNormalisationFactorFilename; 
    
};  
  
} // namespace itk.


#endif /*ITKVelocityFieldDeformableTransformFilename_H_*/
