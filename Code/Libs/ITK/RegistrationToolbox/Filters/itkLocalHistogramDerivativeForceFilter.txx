/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkLocalHistogramDerivativeForceFilter_txx
#define __itkLocalHistogramDerivativeForceFilter_txx

#include "itkLocalHistogramDerivativeForceFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include <itkLogHelper.h>

namespace itk {

template< class TFixedImage, class TMovingImage, class TScalar >
LocalHistogramDerivativeForceFilter< TFixedImage, TMovingImage, TScalar >
::LocalHistogramDerivativeForceFilter()
{
}


template< class TFixedImage, class TMovingImage, class TScalar >
void
LocalHistogramDerivativeForceFilter< TFixedImage, TMovingImage, TScalar >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template< class TFixedImage, class TMovingImage, class TScalar >
void
LocalHistogramDerivativeForceFilter< TFixedImage, TMovingImage, TScalar >
::ThreadedGenerateData(const RegionType& outputRegionForThread, int threadNumber) 
{
  niftkitkDebugMacro(<<"ThreadedGenerateData():Computing histogram force, using Bill Crum's method, thread:" << threadNumber);

  // Initialise.
  double fixedImageEntropy = 0.0;
  double transformedMovingImageEntropy = 0.0;
  double jointEntropy = 0.0;
  double totalFrequency = 0.0;
  std::vector<int> fastFixedImageHistogram;
  std::vector<int> fastTransformedImageHistogram;
  fastFixedImageHistogram.clear();
  fastTransformedImageHistogram.clear();
  
  // Get Histogram
  HistogramPointer histogram = this->GetMetric()->GetHistogram();
  
  // This assumes that the similarity measure has already been run????
  transformedMovingImageEntropy = histogram->EntropyMoving();
  fixedImageEntropy             = histogram->EntropyFixed();
  jointEntropy                  = histogram->JointEntropy();
  totalFrequency                = histogram->GetTotalFrequency();
    
  // Fixed image marginal entropy.
  for (unsigned int i = 0; i < histogram->GetSize()[0]; i++)
  {
    HistogramFrequencyType freq = histogram->GetFrequency(i, 0);
    fastFixedImageHistogram.push_back((int)freq);
  }
  
  // Transformed moving image marginal entropy.
  for (unsigned int i = 0; i < histogram->GetSize()[1]; i++)
  {
    HistogramFrequencyType freq = histogram->GetFrequency(i, 1);
    fastTransformedImageHistogram.push_back((int)freq);
  }
  niftkitkDebugMacro(<<"ThreadedGenerateData():Computed entropies, fixed:" <<  fixedImageEntropy \
      << ", transformedMoving:" << transformedMovingImageEntropy \
      << ", joint:" << jointEntropy \
      << ", totalFrequency=" << totalFrequency); 
  
  // Pointer to output image.
  typename OutputImageType::Pointer forceImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  OutputImageSpacingType spacing = forceImage->GetSpacing();
  
  // Pointer to fixed image.
  typename InputImageType::Pointer fixedImage = 
        static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  // Pointer to transformedMovingImage
  typename InputImageType::Pointer transformedMovingImage = 
        static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  typedef itk::ConstNeighborhoodIterator<InputImageType> NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);

  // Sort out iterators.  
  ImageRegionConstIteratorWithIndex<InputImageType> fixedImageIterator(fixedImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> forceImageIterator(forceImage, outputRegionForThread);  
  NeighborhoodIteratorType transformedImageIterator(radius, transformedMovingImage, outputRegionForThread);
  
  // Set iterators to begining.
  fixedImageIterator.GoToBegin();
  transformedImageIterator.GoToBegin();
  forceImageIterator.GoToBegin();
  
  for (; !fixedImageIterator.IsAtEnd(); ++fixedImageIterator, ++transformedImageIterator, ++forceImageIterator)
  {
    typename OutputImageType::PixelType           forceImageVoxel;
    
    // Check if the voxel is inside the mask. 
    typename TFixedImage::PointType physicalPoint; 
    fixedImage->TransformIndexToPhysicalPoint(fixedImageIterator.GetIndex(), physicalPoint); 
    if (this->m_FixedImageMask && !this->m_FixedImageMask->IsInside(physicalPoint))
    {
      for (int dimensinIndex = 0; dimensinIndex < TFixedImage::ImageDimension; dimensinIndex++)
        forceImageVoxel[dimensinIndex] = 0.0;  
      forceImageIterator.Set(forceImageVoxel);
      continue; 
    }
    
    const typename TFixedImage::PixelType fixedImageVoxel = fixedImageIterator.Get();
    typename HistogramType::MeasurementVectorType fixedImageSample;
    typename HistogramType::IndexType             fixedImageHistogramIndex; 
    typename HistogramType::FrequencyType         fixedImageHistogramFrequency;
    
    fixedImageSample[0] = fixedImageVoxel;
    fixedImageSample[1] = 0;
    
    if (histogram->GetIndex(fixedImageSample, fixedImageHistogramIndex))
    {
      fixedImageHistogramFrequency = fastFixedImageHistogram[fixedImageHistogramIndex[0]];
    }
    else
    {
      fixedImageHistogramFrequency = 0;
    }
    forceImageVoxel = forceImageIterator.Get();

/*
    std::cerr << "fixedImageSample:" << fixedImageSample \
      << ", gives fixedImageHistogramIndex:" << fixedImageHistogramIndex \
      << ", fixedImageHistogramFrequency:" << fixedImageHistogramFrequency << std::endl;
*/
    
    for (int dimensinIndex = 0; dimensinIndex < TFixedImage::ImageDimension; dimensinIndex++)
    {
      typename HistogramType::FrequencyType transformedMovingImageMinusHistogramIndexFrequency;
      typename HistogramType::FrequencyType transformedMovingImageMinusHistogramIndexJointFrequency;
      typename HistogramType::FrequencyType transformedMovingImagePlusHistogramIndexFrequency;
      typename HistogramType::FrequencyType transformedMovingImagePlusHistogramIndexJointFrequency;
      
      typename NeighborhoodIteratorType::OffsetType minusOffset;
      typename NeighborhoodIteratorType::OffsetType plusOffset;
      
      minusOffset.Fill(0);
      plusOffset.Fill(0);
      
      minusOffset[dimensinIndex] += -1;
      plusOffset[dimensinIndex] += 1;
      
/*      
      std::cerr << "minusOffset:" << minusOffset << ", plusOffset:" << plusOffset << std::endl;
*/

      typename TFixedImage::PixelType               transformedMovingImageVoxelMinus = transformedImageIterator.GetPixel(minusOffset);
      typename HistogramType::MeasurementVectorType transformedMovingImageSampleMinus;
      typename HistogramType::MeasurementVectorType transformedMovingImageJointSampleMinus;
      
      typename HistogramType::IndexType             transformedMovingImageMinusJointHistogramIndex; 
      
      transformedMovingImageJointSampleMinus[0] = fixedImageVoxel; 
      transformedMovingImageJointSampleMinus[1] = transformedMovingImageVoxelMinus;
      
      if (histogram->GetIndex(transformedMovingImageJointSampleMinus, transformedMovingImageMinusJointHistogramIndex))
      {
        // Get the frequency at the -ve offset.
        transformedMovingImageMinusHistogramIndexFrequency = fastTransformedImageHistogram[transformedMovingImageMinusJointHistogramIndex[1]];
      }
      else
      {
        transformedMovingImageMinusHistogramIndexFrequency = 0; 
      }
      
      // Get the joint frequency at the -ve offset.
      transformedMovingImageMinusHistogramIndexJointFrequency = histogram->GetFrequency(transformedMovingImageMinusJointHistogramIndex);

/*
      std::cerr << "transformedMovingImageJointSampleMinus:" << transformedMovingImageJointSampleMinus \
        << ", transformedMovingImageMinusJointHistogramIndex:" << transformedMovingImageMinusJointHistogramIndex \
        << ", transformedMovingImageMinusHistogramIndexFrequency:" << transformedMovingImageMinusHistogramIndexFrequency \
        << ",  transformedMovingImageMinusHistogramIndexJointFrequency:" << transformedMovingImageMinusHistogramIndexJointFrequency \
        << std::endl;
*/
      
      typename TFixedImage::PixelType transformedMovingImageVoxelPlus = transformedImageIterator.GetPixel(plusOffset); 
      typename HistogramType::MeasurementVectorType transformedMovingImageSamplePlus;
      typename HistogramType::MeasurementVectorType transformedMovingImageJointSamplePlus;
      typename HistogramType::IndexType transformedMovingImagePlusJointHistogramIndex;
      
      transformedMovingImageJointSamplePlus[0] = fixedImageVoxel; 
      transformedMovingImageJointSamplePlus[1] = transformedMovingImageVoxelPlus;
      
      if (histogram->GetIndex(transformedMovingImageJointSamplePlus, transformedMovingImagePlusJointHistogramIndex))
      {
        // Frequency at the +ve offset.
        transformedMovingImagePlusHistogramIndexFrequency = fastTransformedImageHistogram[transformedMovingImagePlusJointHistogramIndex[1]];
      }
      else
      {
        transformedMovingImagePlusHistogramIndexFrequency = 0; 
      }      
      // Get the joint frequency at the +ve offset.
      transformedMovingImagePlusHistogramIndexJointFrequency = histogram->GetFrequency(transformedMovingImagePlusJointHistogramIndex);

/*
      std::cerr << "transformedMovingImageJointSamplePlus:" << transformedMovingImageJointSamplePlus \
        << ", transformedMovingImagePlusJointHistogramIndex:" << transformedMovingImagePlusJointHistogramIndex \
        << ", transformedMovingImagePlusHistogramIndexFrequency:" << transformedMovingImagePlusHistogramIndexFrequency \
        << ",  transformedMovingImagePlusHistogramIndexJointFrequency:" << transformedMovingImagePlusHistogramIndexJointFrequency \
        << std::endl;
      
      std::cerr << "dimensinIndex: " << dimensinIndex << ", freq: " << transformedMovingImageMinusHistogramIndexFrequency << ","
                            << transformedMovingImageMinusHistogramIndexJointFrequency << ","
                            << transformedMovingImagePlusHistogramIndexFrequency << ","
                            << transformedMovingImagePlusHistogramIndexJointFrequency << std::endl;
*/

      // Calculate the registration force.       
      if (transformedMovingImageMinusHistogramIndexFrequency > 0 && 
          transformedMovingImageMinusHistogramIndexJointFrequency > 0 &&
          transformedMovingImagePlusHistogramIndexFrequency > 0 &&
          transformedMovingImagePlusHistogramIndexJointFrequency > 0)
      {
        forceImageVoxel[dimensinIndex] = ComputeForcePerVoxel(static_cast<double>(totalFrequency),
                                                              static_cast<double>(jointEntropy), 
                                                              static_cast<double>(fixedImageEntropy), 
                                                              static_cast<double>(transformedMovingImageEntropy), 
                                                              static_cast<double>(transformedMovingImageMinusHistogramIndexJointFrequency),
                                                              static_cast<double>(transformedMovingImagePlusHistogramIndexJointFrequency),
                                                              static_cast<double>(transformedMovingImageMinusHistogramIndexFrequency),
                                                              static_cast<double>(transformedMovingImagePlusHistogramIndexFrequency)); 
      }
      else
      {
        forceImageVoxel[dimensinIndex] = 0.0;  
      }
    }
    
    // Here we set the force image vector.
    if (this->GetScaleToSizeOfVoxelAxis())
      {
        for (int dimensinIndex = 0; dimensinIndex < TFixedImage::ImageDimension; dimensinIndex++)
          { 
            forceImageVoxel[dimensinIndex] *= spacing[dimensinIndex]; 
          }      
      }
    forceImageIterator.Set(forceImageVoxel);
  }
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Computing histogram force, using Bill Crum's method, thread:" << threadNumber << ", DONE");
}

} // end namespace itk

#endif
