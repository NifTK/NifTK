/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 Copyright (c) UCL : See the licence file in the top level 
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKDBCImageFilter_TXX_
#define ITKDBCImageFilter_TXX_

#include "itkImageDuplicator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLogImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkExpImageFilter.h"
#include "itkDivideByConstantImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkDivideImageFilter.h"

namespace itk 
{

template <class TImageType, class TMaskType>
DBCImageFilter<TImageType, TMaskType>
::DBCImageFilter()
{
  // need to work on the whole image for fluid registraion. 
  this->m_InputRegionExpansion = 100; 
  this->m_InputRadius = 5; 
  this->m_InputMode = 1; 
}

template <class TImageType, class TMaskType>
void
DBCImageFilter<TImageType, TMaskType>
::GenerateData()
{
  // Calculate the bias fields. 
  CalculateBiasFields(); 
  
  // Apply them. 
  ApplyBiasFields(); 
}


template <class TImageType, class TMaskType>
void
DBCImageFilter<TImageType, TMaskType>
::CalculateBiasFields()
{
  typedef ImageRegionConstIteratorWithIndex<TImageType> ImageConstIterator;
  typedef ImageRegionIteratorWithIndex<TImageType> ImageIterator;
  typedef ImageRegionConstIteratorWithIndex<TMaskType> MaskConstIterator;
  typedef LogImageFilter<TImageType, TImageType> LogImageFilterType; 
  typedef SubtractImageFilter<TImageType, TImageType> SubtractImageFilterType; 
  typedef MedianImageFilter<TImageType, TImageType> MedianImageFilterType; 
  typedef ExpImageFilter<TImageType, TImageType> ExpImageFilterType; 
  typedef DivideByConstantImageFilter<TImageType, typename TImageType::PixelType, TImageType> DivideByConstantImageFilterType; 
  typedef MultiplyImageFilter<TImageType, TImageType> MultiplyImageFilterType; 
  typedef ImageDuplicator<TImageType> DuplicatorType; 
  
  // Quick check before moving on. 
  InputSanityCheck(); 
  int numberOfImages = this->m_InputImages.size(); 
  
  // Rectangular region covering the union of all the masks. Initialise it to max
  typename TImageType::RegionType requestedRegion;
  for (unsigned int i=0; i<TImageType::ImageDimension; i++)
  {
    requestedRegion.SetIndex(i, std::numeric_limits<int>::max());
    requestedRegion.SetSize(i, 0);
  }
  std::cerr << "requestedRegion=" << requestedRegion << std::endl;
      
  // Normalisation. Scale the images to the geometric mean of the mean brain intensities of the images. 
  std::cerr << "Normalising..." << std::endl; 
  long double* mean = new long double[numberOfImages];
  for (int i = 0; i < numberOfImages; i++)
  {
    long double count1 = 0.0; 
    mean[i] = 0.0; 
    ImageConstIterator image1It(this->m_InputImages[i], this->m_InputImages[i]->GetLargestPossibleRegion());
    MaskConstIterator mask1It(this->m_InputImageMasks[i], this->m_InputImageMasks[i]->GetLargestPossibleRegion());; 
    for (image1It.GoToBegin(), mask1It.GoToBegin(); 
        !image1It.IsAtEnd(); 
        ++image1It, ++mask1It)
    {
      if (mask1It.Get() > 0)
      {
        mean[i] += image1It.Get(); 
        count1++; 

        for (unsigned int i=0; i<TImageType::ImageDimension; i++)
        {
          if (requestedRegion.GetIndex(i) > mask1It.GetIndex()[i])
            requestedRegion.SetIndex(i, mask1It.GetIndex()[i]);
          if (requestedRegion.GetIndex(i)+static_cast<int>(requestedRegion.GetSize(i))-1 < mask1It.GetIndex()[i])
            requestedRegion.SetSize(i, mask1It.GetIndex()[i]-requestedRegion.GetIndex(i)+1);
        }
      }
    }
    mean[i] /= count1; 
  }
  std::cerr << "requestedRegion=" << requestedRegion << std::endl;
  typename TImageType::RegionType largestRegion = this->m_InputImages[0]->GetLargestPossibleRegion();
  for (unsigned int i=0; i<TImageType::ImageDimension; i++)
  {
    requestedRegion.SetIndex(i, std::max(requestedRegion.GetIndex(i)-this->m_InputRegionExpansion, 0L));
    requestedRegion.SetSize(i, std::min(requestedRegion.GetSize(i)+2*this->m_InputRegionExpansion, largestRegion.GetSize(i)-requestedRegion.GetIndex(i)));
  }
  std::cerr << "expanded requestedRegion=" << requestedRegion << std::endl;

  double normalisedMean = 0.0;
  for (int i = 0; i < numberOfImages; i++)
  { 
    std::cerr << "Mean[" << i << "]=" << mean[i] << std::endl;
    if (mean[i] <= 0.0)
    {
      std::cerr << "Mean intensity of input image " << i << " is -ve:" << mean[i] << std::endl;
      itkExceptionMacro("Sorry. I cannot handle images with negative mean intensity.")
    }
    normalisedMean += vcl_log(mean[i]);
  }
  normalisedMean /= numberOfImages;
  normalisedMean = vcl_exp(normalisedMean);
  std::cerr << "normalisedMean=" << normalisedMean << std::endl;

  typename DuplicatorType::Pointer* normalisedInputImages = new typename DuplicatorType::Pointer[numberOfImages]; 
  for (int i = 0; i < numberOfImages; i++)
  {
    normalisedInputImages[i] = DuplicatorType::New(); 
    normalisedInputImages[i]->SetInputImage(this->m_InputImages[i]); 
    normalisedInputImages[i]->Update();
    
    ImageIterator image1It(normalisedInputImages[i]->GetOutput(), normalisedInputImages[i]->GetOutput()->GetLargestPossibleRegion());
    for (image1It.GoToBegin(); !image1It.IsAtEnd(); ++image1It)
    {
      // Actually there is no need to normalise. 
      // double normalisedValue = normalisedMean*image1It.Get()/mean[i];
      double normalisedValue = image1It.Get();  
      if (normalisedValue < 1.0)
        normalisedValue = 1.0; 
      image1It.Set(normalisedValue); 
    }
  }
  
  // Take log. 
  typename LogImageFilterType::Pointer* logImageFilter = new typename LogImageFilterType::Pointer[numberOfImages]; 
  
  for (int i = 0; i < numberOfImages; i++)
  {
    std::cerr << "Taking log..." << i << std::endl; 
    logImageFilter[i] = LogImageFilterType::New(); 
    logImageFilter[i]->SetInput(normalisedInputImages[i]->GetOutput()); 
    logImageFilter[i]->Update(); 
  }
  
  // For all the pairs of images, subtract the log images. 
  typename SubtractImageFilterType::Pointer* subtractImageFilter = new typename SubtractImageFilterType::Pointer[numberOfImages*numberOfImages]; 
  for (int i = 0; i < numberOfImages; i++)
  {
    for (int j = i+1; j < numberOfImages; j++)
    {
      std::cerr << "Subtracting..." << i << "," << j << std::endl; 
      int index = i*numberOfImages+j; 
      subtractImageFilter[index] = SubtractImageFilterType::New(); 
      subtractImageFilter[index]->SetInput1(logImageFilter[i]->GetOutput()); 
      subtractImageFilter[index]->SetInput2(logImageFilter[j]->GetOutput()); 
      // Just need consecutive pairwise differential bias fields for mode 2. 
      if (this->m_InputMode == 2)
        break; 
    }
  }
  
  typename TImageType::Pointer* pairwiseBiasRatioImage = new typename TImageType::Pointer[numberOfImages*numberOfImages];
  
  // Apply median filter to the subtraction images to obtain the bias.  
  typename MedianImageFilterType::Pointer* medianImageFilter = new typename MedianImageFilterType::Pointer[numberOfImages*numberOfImages];
  typename TImageType::SizeType kernelRadius; 
  kernelRadius.Fill(this->m_InputRadius); 
  for (int i = 0; i < numberOfImages; i++)
  {
    for (int j = i+1; j < numberOfImages; j++)
    {
      std::cerr << "Applying median filter..." << i << "," << j << std::endl; 
      int index = i*numberOfImages+j; 
      medianImageFilter[index] = MedianImageFilterType::New(); 
      medianImageFilter[index]->SetInput(subtractImageFilter[index]->GetOutput()); 
      medianImageFilter[index]->SetRadius(kernelRadius); 
      medianImageFilter[index]->GetOutput()->SetRequestedRegion(requestedRegion);
      medianImageFilter[index]->Update(); 
      pairwiseBiasRatioImage[index] = medianImageFilter[index]->GetOutput(); 
      // Just need consecutive pairwise differential bias fields for mode 2. 
      if (this->m_InputMode == 2)
        break; 
    }
  }
  
  typename DuplicatorType::Pointer* duplicator = new typename DuplicatorType::Pointer[numberOfImages*numberOfImages]; 
  // Compose the non-consecutive pairwise differential bias fields form the consecutive ones. 
  if (this->m_InputMode == 2)
  {
    for (int i = 0; i < numberOfImages; i++)
    {
      for (int j = i+2; j < numberOfImages; j++)
      {
        std::cerr << "Composing non-consecutive differential bias field..." << i << "," << j << std::endl; 
        int nonConsetiveTimePointIndex = i*numberOfImages+j; 
        duplicator[nonConsetiveTimePointIndex] = DuplicatorType::New(); 
        duplicator[nonConsetiveTimePointIndex]->SetInputImage(normalisedInputImages[i]->GetOutput()); 
        duplicator[nonConsetiveTimePointIndex]->Update();
        pairwiseBiasRatioImage[nonConsetiveTimePointIndex] = duplicator[nonConsetiveTimePointIndex]->GetOutput(); 
        pairwiseBiasRatioImage[nonConsetiveTimePointIndex]->FillBuffer(0.0); 
        for (int m = i, n=i+1; m < j; m++,n++)
        {
          std::cerr << "  Adding in..." << m << "," << n << std::endl; 
          int consetiveTimePointIndex = m*numberOfImages+n; 
          ImageIterator image1It(pairwiseBiasRatioImage[nonConsetiveTimePointIndex], requestedRegion);
          ImageIterator image2It(medianImageFilter[consetiveTimePointIndex]->GetOutput(), requestedRegion);
          for (image1It.GoToBegin(), image2It.GoToBegin(); 
              !image1It.IsAtEnd(); 
              ++image1It, ++image2It)
          {
            image1It.Set(image1It.Get()+image2It.Get()); 
          }
        }
      }
    }
  }
  
  // Calculate the bias ratio images in log scale, i.e. R_1 = (r_12*r_13)^(1/3), R_2=c(r_21*r_23)^(1/3), ... etc. 
  typename DuplicatorType::Pointer* biasRatioImage = new typename DuplicatorType::Pointer[numberOfImages]; 
  for (int i = 0; i < numberOfImages; i++)
  {
    biasRatioImage[i] = DuplicatorType::New(); 
    biasRatioImage[i]->SetInputImage(normalisedInputImages[i]->GetOutput()); 
    biasRatioImage[i]->Update();
    biasRatioImage[i]->GetOutput()->FillBuffer(0.0); 
    for (int j = 0; j < numberOfImages; j++)
    {
      float sign = 1.0; 
      int index = i*numberOfImages+j; 
      if (i > j)   
      {
        sign = -1.0; 
        index = j*numberOfImages+i; 
      }
      if (i != j)
      {
        std::cerr << "Calculating bias ratio in log scale..." << i << "," << j << std::endl; 
        ImageIterator image1It(biasRatioImage[i]->GetOutput(), requestedRegion);
        ImageIterator image2It(pairwiseBiasRatioImage[index], requestedRegion);
        for (image1It.GoToBegin(), image2It.GoToBegin(); 
            !image1It.IsAtEnd(); 
            ++image1It, ++image2It)
        {
          image1It.Set(image1It.Get()+(sign*image2It.Get())/numberOfImages); 
        }
      }
    }
  }
  
  typename ExpImageFilterType::Pointer* expImageFilter = new typename ExpImageFilterType::Pointer[numberOfImages]; 
  this->m_BiasFields.clear(); 
  for (int i = 0; i < numberOfImages; i++)
  {
    // Exponential the the bias ratio. 
    expImageFilter[i] = ExpImageFilterType::New(); 
    expImageFilter[i]->SetInput(biasRatioImage[i]->GetOutput()); 
    expImageFilter[i]->Update(); 
    this->m_BiasFields.push_back(expImageFilter[i]->GetOutput()); 
    this->m_BiasFields[i]->DisconnectPipeline(); 
  }
  
  delete [] mean;
  delete [] logImageFilter; 
  delete [] medianImageFilter; 
  delete [] biasRatioImage; 
  delete [] expImageFilter; 
  delete [] pairwiseBiasRatioImage; 
  delete [] duplicator; 
}

template <class TImageType, class TMaskType>
void
DBCImageFilter<TImageType, TMaskType>
::ApplyBiasFields()
{
  InputSanityCheck(); 
  
  typedef DivideImageFilter<TImageType, TImageType, TImageType> DivideImageFilterType; 
  typename DivideImageFilterType::Pointer* divideImageFilter = new typename DivideImageFilterType::Pointer[this->m_InputImages.size()]; 
  
  // Apply the bias to the images by dividing by the bias ratio. 
  for (unsigned int i = 0; i < this->m_InputImages.size(); i++)
  {
    divideImageFilter[i] = DivideImageFilterType::New(); 
    divideImageFilter[i]->SetInput1(this->m_InputImages[i]); 
    divideImageFilter[i]->SetInput2(this->m_BiasFields[i]); 
    divideImageFilter[i]->Update(); 
    this->m_OutputImages.push_back(divideImageFilter[i]->GetOutput()); 
    this->m_OutputImages[i]->DisconnectPipeline(); 
  }
  delete [] divideImageFilter; 
}
    


}

#endif /*ITKDBCImageFilter_TXX_*/
