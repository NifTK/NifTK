/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKSimpleFuzzyCMeansClusteringImageFilter_TXX_
#define ITKSimpleFuzzyCMeansClusteringImageFilter_TXX_

#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

namespace itk 
{

template <class TInputImage, class TInputMask>
void
SimpleFuzzyCMeansClusteringImageFilter<TInputImage, TInputMask>
::GenerateData()
{
  typename TInputImage::ConstPointer inputImage = this->GetInput();
  itk::ImageRegionConstIterator<TInputImage> inputImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<TInputMask> inputMaskIterator(this->m_InputMask, this->m_InputMask->GetLargestPossibleRegion());
  unsigned int numberOfVoxels = 0; 

  {
    // Build up the sample. 
    SampleType::Pointer inputSample = SampleType::New() ;
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
        numberOfVoxels++; 
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
  }
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
    std::cout << "kmeans=" << this->m_FinalMeans[classIndex] << "," << this->m_FinalStds[classIndex] << std::endl;
  }
  
  std::cout << "size=" << numberOfVoxels << std::endl; 
  typename MeasurementVectorType::ValueType* data = new typename MeasurementVectorType::ValueType[numberOfVoxels]; 
  typename MeasurementVectorType::ValueType* fuzzyMembership = new typename MeasurementVectorType::ValueType[m_NumberOfClasses*numberOfVoxels]; 
  bool isConverged = false; 
  unsigned int numberOfIterations = 0; 
  
  // Build up the sample. 
  unsigned int voxelCount = 0; 
  inputImageIterator.GoToBegin();
  inputMaskIterator.GoToBegin();
  for ( ; !inputImageIterator.IsAtEnd(); ++inputImageIterator, ++inputMaskIterator )
  {
    if (inputMaskIterator.Get() > 0)
    {
      typename TInputImage::PixelType value = inputImageIterator.Get();

      // TODO: quick hack - remove the -ve number from ITK transformed images. 
      if (value < 0)
        value = 0;
      data[voxelCount] = static_cast<typename MeasurementVectorType::ValueType>(value);
      voxelCount++; 
    }
  }
  
  
  while (!isConverged)
  {
    ParametersType oldMeans;
    
    oldMeans.SetSize(this->m_NumberOfClasses);
    for (unsigned int i = 0; i < m_NumberOfClasses; i++)
    {
      oldMeans[i] = m_FinalMeans[i]; 
      //std::cout << "mean[" << i << "]=" << oldMeans[i] << std::endl; 
    }
    
    // 1. Calculate the fuzzy membership. 
    for (unsigned int k = 0; k < numberOfVoxels; k++)
    {
      for (unsigned int i = 0; i < m_NumberOfClasses; i++)
      {
        typename MeasurementVectorType::ValueType x_k = data[k]; 
        typename MeasurementVectorType::ValueType v_i = m_FinalMeans[i]; 
        
        fuzzyMembership[i*numberOfVoxels+k] = 0.0; 
        for (unsigned int j = 0; j < m_NumberOfClasses; j++)
        {
          typename MeasurementVectorType::ValueType v_j = m_FinalMeans[j]; 
        
          fuzzyMembership[i*numberOfVoxels+k] += pow((double)(fabs(x_k-v_i)/fabs(x_k-v_j)), 2.0/(m_Fuzziness-1.0)); 
        }
        
        fuzzyMembership[i*numberOfVoxels+k] = 1.0/fuzzyMembership[i*numberOfVoxels+k]; 
      }
    }
    
    // 2. Calculate the centres. 
    for (unsigned int i = 0; i < m_NumberOfClasses; i++)
    {
      typename MeasurementVectorType::ValueType total = 0.0; 
      typename MeasurementVectorType::ValueType weight = 0.0; 
      
      for (unsigned int k = 0; k < numberOfVoxels; k++)
      {
        typename MeasurementVectorType::ValueType x_k = data[k]; 
        
        total += pow(fuzzyMembership[i*numberOfVoxels+k], m_Fuzziness)*x_k; 
        weight += pow(fuzzyMembership[i*numberOfVoxels+k], m_Fuzziness); 
      }
      //std::cout << "total=" << total << ",weight=" << weight << std::endl; 
      m_FinalMeans[i] = total/weight; 
    }
    
    // 3. Check for convergence. 
    for (unsigned int i = 0; i < m_NumberOfClasses; i++)
    {
      isConverged = true; 
      if (fabs(oldMeans[i] - m_FinalMeans[i]) > 0.0001)
        isConverged = false; 
    }
    numberOfIterations++; 
  } 
  
  std::cout << "Number of iterations=" << numberOfIterations << std::endl; 
  
  typedef ImageFileWriter<TOutputImage> WriterType;
  typename WriterType::Pointer writer = WriterType::New(); 
  
  // Saving the fuzzy memebership. 
  for (unsigned int i = 0; i < m_NumberOfClasses; i++)
  {
    voxelCount = 0; 
    for (outputImageIterator.GoToBegin(), inputMaskIterator.GoToBegin(); 
        !outputImageIterator.IsAtEnd(); 
        ++outputImageIterator, ++inputMaskIterator)
    {
      if (inputMaskIterator.Get() > 0)
      {
        outputImageIterator.Set(fuzzyMembership[i*numberOfVoxels+voxelCount]); 
        voxelCount++; 
      }
      else
      {
        outputImageIterator.Set(0.0); 
      }
    }
    char outputFilename[4096+1]; 
    
    sprintf(outputFilename, this->m_OutputFileNameFormat.c_str(), i); 
    writer->SetFileName(outputFilename); 
    writer->SetInput(outputImage); 
    writer->Update(); 
  }
  
  delete[] fuzzyMembership; 
}


}

#endif
