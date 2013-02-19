/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKSIMPLEKMEANSCLUSTERINGIMAGEFILTER_TXX_
#define ITKSIMPLEKMEANSCLUSTERINGIMAGEFILTER_TXX_

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

namespace itk 
{

template <class TInputImage, class TInputMask, class TOutputImage>
void
SimpleKMeansClusteringImageFilter<TInputImage, TInputMask, TOutputImage>
::GenerateData()
{
  typename TInputImage::ConstPointer inputImage = this->GetInput();
  itk::ImageRegionConstIterator<TInputImage> inputImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<TInputMask> inputMaskIterator(this->m_InputMask, this->m_InputMask->GetLargestPossibleRegion());
  SampleType::Pointer inputSample = SampleType::New() ;

  // Build up the sample. 
  inputImageIterator.GoToBegin();
  inputMaskIterator.GoToBegin();
  for ( ; !inputImageIterator.IsAtEnd(); ++inputImageIterator, ++inputMaskIterator )
  {
    if (inputMaskIterator.Get() > 0)
    {
      typename TInputImage::PixelType value = inputImageIterator.Get();
      MeasurementVectorType oneSample;

      // TODO: quick hack - remove the -ve number from ITK transformed images. 
      if (value < 0)
        value = 0;
      oneSample[0] = static_cast<double>(value);
      inputSample->PushBack(oneSample);
    }
  }
  
  TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();

  // Prepare the K-d tree. 
  treeGenerator->SetSample(inputSample);
  treeGenerator->SetBucketSize(16);
  treeGenerator->Update();

  typename EstimatorType::Pointer estimator = EstimatorType::New();

  // K-Means clustering. 
  estimator->SetParameters(this->m_InitialMeans);
  estimator->SetKdTree(treeGenerator->GetOutput());
  estimator->SetMaximumIteration(500);
  estimator->SetCentroidPositionChangesThreshold(0.0);
  estimator->StartOptimization();

  this->m_FinalMeans = estimator->GetParameters();
  this->m_FinalStds.SetSize(this->m_NumberOfClasses);
  this->m_FinalClassSizes.SetSize(this->m_NumberOfClasses);
  for (unsigned int classIndex = 0; classIndex < this->m_NumberOfClasses; classIndex++)
  {
    this->m_FinalStds[classIndex] = 0.0;
    this->m_FinalClassSizes[classIndex] = 0.0;
  }

  // Allocate the output image. 
  typename TOutputImage::Pointer outputImage = this->GetOutput();
  
  outputImage->SetRequestedRegion(this->GetInput()->GetLargestPossibleRegion());
  this->AllocateOutputs();
  
  itk::ImageRegionIterator<TOutputImage> outputImageIterator(outputImage, outputImage->GetLargestPossibleRegion());
  
  // Classify each voxel according the distance to the means. 
  inputImageIterator.GoToBegin();
  inputMaskIterator.GoToBegin();
  outputImageIterator.GoToBegin();
  this->m_RSS = 0.0; 
  this->m_NumberOfSamples = 0.0; 
  for ( ; !inputImageIterator.IsAtEnd(); ++inputImageIterator, ++inputMaskIterator, ++outputImageIterator )
  {
    if (inputMaskIterator.Get() > 0)
    {
      typename TInputImage::PixelType value = inputImageIterator.Get();
      int bestClass = -1;
      double bestDistance = std::numeric_limits<double>::max();
      
      // TODO: quick hack - remove the -ve number from ITK transformed images. 
      if (value < 0)
        value = 0;
      for (unsigned int classIndex = 0; classIndex < this->m_NumberOfClasses; classIndex++ )
      {
        double currentDistance = fabs(value-this->m_FinalMeans[classIndex]);

        if ( currentDistance < bestDistance )
        {
          bestDistance = currentDistance;
          bestClass = classIndex;
        }
      }
      this->m_RSS += bestDistance*bestDistance; 
      this->m_NumberOfSamples++;
      this->m_FinalStds[bestClass] = this->m_FinalStds[bestClass]+(this->m_FinalMeans[bestClass]-value)*(this->m_FinalMeans[bestClass]-value);
      this->m_FinalClassSizes[bestClass] = this->m_FinalClassSizes[bestClass]+1;
      outputImageIterator.Set(bestClass+1);
    }
    else
    {
      outputImageIterator.Set(0);
    }
  }
  
  for (unsigned int classIndex = 0; classIndex < this->m_NumberOfClasses; classIndex++ )
  {
    this->m_FinalStds[classIndex] = sqrt(this->m_FinalStds[classIndex]/this->m_FinalClassSizes[classIndex]);
    //std::cout << this->m_FinalMeans[classIndex] << std::endl << this->m_FinalStds[classIndex] << std::endl;
  }
}


}

#endif
