/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBASICIMAGECOMPARISONFUNCTIONS_H_
#define ITKBASICIMAGECOMPARISONFUNCTIONS_H_

namespace itk
{

/**
 * Check if the two images have the same size. 
 * \param typename TInputImage1::Pointer image1 The first image. 
 * \param typename TInputImage2::Pointer image2 The second image. 
 * \return true if the two images have the same size, false if otherwise. 
 */
template <class TInputImage1, class TInputImage2> 
ITK_EXPORT bool IsSameRegionSize(typename TInputImage1::Pointer image1, typename TInputImage2::Pointer image2)
{
  image1->Update();
  image2->Update();
  
  if (image1->GetLargestPossibleRegion().GetSize() == image2->GetLargestPossibleRegion().GetSize())
    return true; 
  
  return false; 
}

/**
 * Check if the two images have the same voxel sizes, within a relative error of 0.00005 in each dimension. 
 * \param typename TInputImage1::Pointer image1 The first image. 
 * \param typename TInputImage2::Pointer image2 The second image. 
 * \return true if the two images have the same voxel size, false if otherwise. 
 */
template <class TInputImage1, class TInputImage2>
ITK_EXPORT bool IsSameVoxelSize(typename TInputImage1::Pointer image1, typename TInputImage2::Pointer image2)
{
  image1->Update();
  image2->Update();
  
  typename TInputImage1::SpacingType samplingSpacing1 = image1->GetSpacing();
  typename TInputImage1::SpacingType::ConstIterator samplingSpacingIterator1 = samplingSpacing1.Begin(); 
  typename TInputImage2::SpacingType samplingSpacing2 = image2->GetSpacing();
  typename TInputImage2::SpacingType::ConstIterator samplingSpacingIterator2 = samplingSpacing2.Begin();
  
  // Random max relative error allowed. 
  const double maxRelativeError = 0.00005;
  
  if (samplingSpacing1.Size() != samplingSpacing2.Size())
    return false; 
  
  for (samplingSpacingIterator1 = samplingSpacing1.Begin(), samplingSpacingIterator2 = samplingSpacing2.Begin();
       samplingSpacingIterator1 != samplingSpacing1.End();
       ++samplingSpacingIterator1, ++samplingSpacingIterator2)
  {
    double spacing1 = static_cast<double>(*samplingSpacingIterator1);
    double spacing2 = static_cast<double>(*samplingSpacingIterator2);
    
    if (spacing1 != spacing2)
    {
      double relativeError = fabs(spacing1-spacing2)/
                             std::max(fabs(spacing1), fabs(spacing2));  
      
      if (relativeError > maxRelativeError)
        return false; 
    }
  }
  
  return true; 
}

}
#endif /*ITKBASICIMAGECOMPARISONFUNCTIONS_H_*/
