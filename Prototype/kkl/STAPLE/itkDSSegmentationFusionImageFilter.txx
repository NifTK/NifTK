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
#ifndef __itkDSSegmentationFusionImageFilter_txx
#define __itkDSSegmentationFusionImageFilter_txx

#include <algorithm>
#include "itkDSSegmentationFusionImageFilter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkMultipleDilateImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSegmentationReliabilityCalculator.h"
#include "itkCastImageFilter.h"

namespace itk
{
  
template< typename TInputImage, typename TOutputImage >
typename DSSegmentationFusionImageFilter< TInputImage,TOutputImage>::InputPixelType
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::ComputeMaximumInputValue()
{
  InputPixelType maxLabel = 0;

  typedef ImageRegionConstIterator< TInputImage > IteratorType;

  // Record the number of input files.
  const unsigned int numberOfInputFiles = this->GetNumberOfInputs();

  for(unsigned int i = 0; i < numberOfInputFiles; ++i)
  {
    const InputImageType* inputImage =  this->GetInput(i);
    IteratorType it(inputImage, inputImage->GetBufferedRegion());
    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      maxLabel = vnl_math_max(maxLabel, it.Get());
    }
  }
  
  this->m_ForegroundLabel = maxLabel; 
  std::cout << "Foreground label = " << (int)m_ForegroundLabel << std::endl; 
  
  // TODO: deal with the labels properly. 
  // Two label only now - assuming background is 0. 
  // background = 0.
  m_LabelToPowerSetIndexMap[0] = 1; 
  // foreground - from the image. 
  m_LabelToPowerSetIndexMap[m_ForegroundLabel] = 2; 
  
  m_PowerSetIndexToLabelMap[1] = 0; 
  m_PowerSetIndexToLabelMap[2] = m_ForegroundLabel; 
  
  if (m_SegmentationReliability.size() == 0)
  {
    for(unsigned int i = 0; i < numberOfInputFiles; ++i)
    {
      m_SegmentationReliability[i] = 1.0; 
    }
  }

  return maxLabel;
}
  
  
template< typename TInputImage, typename TOutputImage >
double
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::GetReliabiltiyFromSimilarityMeasure(int mode, HistogramImageToImageMetricType* metric)
{
  typename TransformType::ParametersType parameters = this->m_IdentityTransform->GetParameters(); 
  
  switch (mode)
  {
    case 0: 
    case 1:
    case 2:
      return pow(fabs(metric->GetValue(parameters)), this->m_Gain); 
  
    case 3:
    case 4:
      return pow(fabs(metric->GetValue(parameters))/2.0, this->m_Gain); 
      
    case 5:
    case 6:
      return fabs(1.0/sqrt(metric->GetValue(parameters))); 
      
    default: 
      itkExceptionMacro("Unknow reliability mode:" << m_ReliabilityMode); 
  }
  
  return 0.0; 
}

template< typename TInputImage, typename TOutputImage >
void
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::NormaliseImage(typename TInputImage::Pointer inputImage, typename TInputImage::ConstPointer mask)
{
  typedef ImageRegionIterator<TInputImage> AtlasIteratorType;
  typedef ImageRegionConstIterator<TInputImage> MaskIteratorType;
  AtlasIteratorType atlasIt(inputImage, inputImage->GetLargestPossibleRegion()); 
  MaskIteratorType maskIt(mask, mask->GetLargestPossibleRegion()); 
  
  double mean = 0.0;
  double voxelCount = 0.0; 
  for (maskIt.GoToBegin(), atlasIt.GoToBegin(); !atlasIt.IsAtEnd(); ++maskIt, ++atlasIt)
  {
    if (maskIt.Get() > 0)
    {
      mean += atlasIt.Get(); 
      voxelCount++; 
    }
  }
  mean /= voxelCount; 
  std::cout << "mean=" << mean << std::endl; 
  for (maskIt.GoToBegin(), atlasIt.GoToBegin(); !atlasIt.IsAtEnd(); ++maskIt, ++atlasIt)
    atlasIt.Set((OutputPixelType)((1000.0*atlasIt.Get())/mean)); 
}

 
template< typename TInputImage, typename TOutputImage >
void
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::BeforeThreadedGenerateData ()
{
  Superclass::BeforeThreadedGenerateData();

  // determine the maximum label in all input images
  this->m_TotalLabelCount = this->ComputeMaximumInputValue() + 1;

  if ( ! this->m_HasLabelForUndecidedPixels )
  {
    this->m_LabelForUndecidedPixels = this->m_TotalLabelCount;
  }
  
  std::cout << "global reliability = "; 
  for (unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
  {
    std::cout << m_SegmentationReliability[i] << " "; 
  }
  std::cout << std::endl; 
  
  // Allocate the output image.
  typename TOutputImage::Pointer output = this->GetOutput();
  output->SetBufferedRegion(output->GetRequestedRegion() );
  output->Allocate();
  
  // Allocate the image to store the conflict. 
  m_ConflictImage = ConflictImageType::New(); 
  m_ConflictImage->SetRegions(output->GetRequestedRegion()); 
  m_ConflictImage->Allocate(); 
  
  // Allocate the consensus image. 
  m_ConsensusImage = TInputImage::New(); 
  m_ConsensusImage->SetRegions(output->GetRequestedRegion()); 
  m_ConsensusImage->Allocate(); 
  
  m_ForegroundBeliefImage = BeliefImageType::New(); 
  m_ForegroundBeliefImage->SetRegions(output->GetRequestedRegion()); 
  m_ForegroundBeliefImage->Allocate(); 
  m_ForegroundPlausibilityImage = BeliefImageType::New(); 
  m_ForegroundPlausibilityImage->SetRegions(output->GetRequestedRegion()); 
  m_ForegroundPlausibilityImage->Allocate(); 
  
  // Normalise the mean intensity of the images to 1000 within the ROIs. 
  if (this->m_TargetImage.IsNotNull())
  {
    NormaliseImage(this->m_TargetImage.GetPointer(), this->GetInput(0)); 
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
      NormaliseImage(this->m_RegisteredAtlases[i].GetPointer(), this->GetInput(i)); 
  }
  
  // Set up the similarity measure.
  if (this->m_ReliabilityMode > 0)
  {   
    this->m_IdentityTransform = TransformType::New(); 
    
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
    {
      this->m_Interpolators.push_back(InterpolatorType::New()); 
      this->m_Interpolators[i]->SetInputImage(this->m_RegisteredAtlases[i]);
    }
    for (unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
    {
      unsigned int nBins = 256;
      typename NormalizedMutualInformationHistogramImageToImageMetricType::HistogramType::SizeType histSize;
      histSize[0] = nBins;
      histSize[1] = nBins;
      typename HistogramImageToImageMetricType::Pointer metric; 
      
      switch (m_ReliabilityMode)
      {
        case 0: 
        case 1:
        case 2:
          metric = CorrelationCoefficientHistogramImageToImageMetricType::New(); 
          break; 
        case 3:
        case 4:
          metric = NormalizedMutualInformationHistogramImageToImageMetricType::New();  
          break; 
        case 5: 
        case 6: 
          metric = MeanSquaresHistogramImageToImageMetricType::New(); 
          break; 
        default:
          itkExceptionMacro("Unknow reliability mode:" << m_ReliabilityMode); 
      }
      this->m_Metrics.push_back(metric); 
      this->m_Metrics[i]->SetFixedImage(this->m_TargetImage);
      this->m_Metrics[i]->SetMovingImage(this->m_RegisteredAtlases[i]);
      this->m_Metrics[i]->SetTransform(this->m_IdentityTransform); 
      this->m_Metrics[i]->SetInterpolator(this->m_Interpolators[i]);
      this->m_Metrics[i]->SetComputeGradient(false); 
      this->m_Metrics[i]->SetHistogramSize(histSize);
      this->m_Metrics[i]->SetFixedImageRegion(this->m_TargetImage->GetLargestPossibleRegion());
      this->m_Metrics[i]->Initialize();
    }
    
    if (this->m_ReliabilityMode == 2 || this->m_ReliabilityMode == 4 || this->m_ReliabilityMode == 6)
    {
      typedef ImageMaskSpatialObject<InputImageDimension> MaskType; 
      typedef MultipleDilateImageFilter<TInputImage> MultipleDilateImageFilterType;
      typedef CastImageFilter<TInputImage, typename MaskType::ImageType> CastImageFilterType; 
      
      for (unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
      {
        typename MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
        typename MaskType::Pointer mask = MaskType::New(); 
        typename CastImageFilterType::Pointer castFilter = CastImageFilterType::New(); 
  
        // Dilate multiple times. 
        multipleDilateImageFilter->SetNumberOfDilations(1);
        multipleDilateImageFilter->SetInput(this->GetInput(i));
        multipleDilateImageFilter->SetDilateValue(this->m_ForegroundLabel); 
        multipleDilateImageFilter->Update(); 
        castFilter->SetInput(multipleDilateImageFilter->GetOutput());
        castFilter->Update(); 
        mask->SetImage(castFilter->GetOutput()); 
        mask->Update(); 
        this->m_Metrics[i]->SetFixedImageMask(mask); 
        m_SegmentationReliability[i] = GetReliabiltiyFromSimilarityMeasure(this->m_ReliabilityMode, this->m_Metrics[i]); 
        //m_SegmentationReliability[i] = GetReliabiltiyFromSimilarityMeasure(this->m_ReliabilityMode, this->m_Metrics[i])*  pow((1.0-((double)i)/((double)this->GetNumberOfInputs()*2.0)), 2); 
        std::cout << "global relaiblity=" << m_SegmentationReliability[i] << std::endl; 
      }
    }
  }      
}



template< typename TInputImage, typename TOutputImage >
void
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::CalculateExpectedSegmentation(const OutputImageRegionType& outputRegionForThread)
{
  typedef ImageRegionConstIterator<TInputImage> IteratorType;
  typedef ImageRegionIterator<TInputImage> ConsensusIteratorType;
  typedef ImageRegionIterator<TOutputImage> OutIteratorType;
  typedef ImageRegionIterator<ConflictImageType> ConflictImageIteratorType;
  typedef ImageRegionIterator<BeliefImageType> BelieveImageIteratorType;
  typedef ImageRegionIterator<BeliefImageType> PlausibilityImageIteratorType;

  typename TOutputImage::Pointer output = this->GetOutput();
  double precision = 0.0; 
  double totalConflict = 0.0; 
  
  // Set up the local region where the similarity measure will be evaluated. 
  typename TInputImage::RegionType localRegion; 
  for (unsigned int i = 0; i < ImageDimension; i++)
    localRegion.SetSize(i, 2*this->m_LocalRegionRadius+1); 
  
  // Record the number of input files.
  const unsigned int numberOfInputFiles = this->GetNumberOfInputs();
  // create and initialize all input image iterators
  IteratorType *it = new IteratorType[numberOfInputFiles];
  for ( unsigned int i = 0; i < numberOfInputFiles; ++i)
  {
    it[i] = IteratorType( this->GetInput(i), outputRegionForThread);
  }
  
  // Find all the voxels with total consensus. 
  OutIteratorType out = OutIteratorType(output, outputRegionForThread);
  ConsensusIteratorType consensusIterator(this->m_ConsensusImage, outputRegionForThread); 
  BelieveImageIteratorType believeIterator(this->m_ForegroundBeliefImage, outputRegionForThread); 
  for (consensusIterator.GoToBegin(), out.GoToBegin(), believeIterator.GoToBegin(); 
       !consensusIterator.IsAtEnd(); 
       ++consensusIterator, ++out, ++believeIterator)
  {
    const InputPixelType label = it[0].Get();
    
    consensusIterator.Set(1); 
    for (unsigned int i = 1; i < numberOfInputFiles; ++i)
    {
      if (label != it[i].Get())
      {
        consensusIterator.Set(0); 
        break;  
      }
    }
    if (consensusIterator.Get() == 1)
    {
      out.Set(label); 
      believeIterator.Set(1.0); 
    }
    
    for (unsigned int i = 0; i < numberOfInputFiles; ++i)
      ++it[i]; 
  }
  
  for ( unsigned int i = 0; i < numberOfInputFiles; ++i)
    it[i].GoToBegin(); 
  ConflictImageIteratorType conflictIterator(this->m_ConflictImage, outputRegionForThread); 
  BelieveImageIteratorType plausibilityIterator(this->m_ForegroundPlausibilityImage, outputRegionForThread); 
  for (out.GoToBegin(), conflictIterator.GoToBegin(), consensusIterator.GoToBegin(), believeIterator.GoToBegin(), plausibilityIterator.GoToBegin(); 
       !out.IsAtEnd(); 
       ++out, ++conflictIterator, ++consensusIterator, ++believeIterator, ++plausibilityIterator)
  {
    // Don't need to think about it if they all agree. 
    if (consensusIterator.Get() == 1)
    {
      for ( unsigned int i = 0; i < numberOfInputFiles; ++i)
        ++it[i]; 
      continue; 
    }
    
    // Believe function. { 0, background, foreground, background or foreground }. 
    std::map<int, double> beliefFunction; 
    // Conflict. 
    double conflict = 0.0; 
    // Current voxel index. 
    typename TOutputImage::IndexType index = out.GetIndex(); 
    // Centre around the current voxel. 
    for (unsigned int i = 0; i < ImageDimension; i++)
      index[i] -= this->m_LocalRegionRadius; 
    localRegion.SetIndex(index); 
    
    // Initialise to be total ignorant about background and foreground. 
    beliefFunction[3] = 1.0; 
    
    // count number of votes for the labels
    for( unsigned int i = 0; i < numberOfInputFiles; ++i)
    {
      const InputPixelType label = it[i].Get();
      // Current believe function.
      typename std::map<int, double> currentBelieveFunction; 
      // New believe function. 
      typename std::map<int, double> combinedBelieveFunction; 
      
      if (this->m_ReliabilityMode > 0)
        this->m_Metrics[i]->SetFixedImageRegion(localRegion);
      switch (this->m_ReliabilityMode)
      {
        case 0: 
        case 2:
        case 4:
        case 6:
          {
            currentBelieveFunction[m_LabelToPowerSetIndexMap[label]] = m_SegmentationReliability[i]; 
            currentBelieveFunction[3] = 1.0-m_SegmentationReliability[i]; 
          }
          break; 
        case 1: 
        case 3: 
        case 5: 
          {
            double localReliability = GetReliabiltiyFromSimilarityMeasure(this->m_ReliabilityMode, this->m_Metrics[i]); 
            //std::cout << "local reliability=" << localReliability << std::endl; 
            currentBelieveFunction[m_LabelToPowerSetIndexMap[label]] = localReliability; 
            currentBelieveFunction[3] = 1.0-localReliability; 
          }
          break; 
        default:
          itkExceptionMacro("Unknown reliability mode:" << this->m_ReliabilityMode); 
      }
      
      // Combine the believe function using Dampster's rule of combination. 
      // TODO: more than 2 labels.
      combinedBelieveFunction[1] = beliefFunction[1]*currentBelieveFunction[1] + 
                                   beliefFunction[1]*currentBelieveFunction[3] + 
                                   beliefFunction[3]*currentBelieveFunction[1]; 
      
      combinedBelieveFunction[2] = beliefFunction[2]*currentBelieveFunction[2] + 
                                   beliefFunction[2]*currentBelieveFunction[3] +
                                   beliefFunction[3]*currentBelieveFunction[2]; 
      
      combinedBelieveFunction[3] = beliefFunction[3]*currentBelieveFunction[3]; 
      
      double normK = 1.0 - beliefFunction[1]*currentBelieveFunction[2] - beliefFunction[2]*currentBelieveFunction[1]; 
      
      if (this->m_CombinationMode == 0)
      {
        // Dampster-Shafter - redistribute the conflict. 
        combinedBelieveFunction[0] = 0.0; 
        combinedBelieveFunction[1] /= normK; 
        combinedBelieveFunction[2] /= normK; 
        combinedBelieveFunction[3] /= normK; 
      }
      else if (this->m_CombinationMode == 1)
      {
        // Transferable belief model - conflict is assigned to the empty set. 
        combinedBelieveFunction[0] = beliefFunction[1]*currentBelieveFunction[2] + 
                                     beliefFunction[2]*currentBelieveFunction[1] + 
                                     beliefFunction[0]*currentBelieveFunction[0]; 
      }
      
      beliefFunction = combinedBelieveFunction; 
      conflict += log(1/normK); 
      
      ++(it[i]);
      
    }
    
    //std::cout << std::endl; 
    //std::cout << "believe function=" << beliefFunction[1] << "," << beliefFunction[2] << std::endl; 
    
    if (this->m_CombinationMode == 0)
    {
      if (beliefFunction[2] - beliefFunction[1] > 0.0001 && beliefFunction[2]+beliefFunction[3] > this->m_PlausibilityThreshold)
      {
        out.Set(m_PowerSetIndexToLabelMap[2]); 
        precision += (1.0-beliefFunction[1]) - beliefFunction[2]; 
      }
      else if (beliefFunction[1] - beliefFunction[2] > 0.0001 && beliefFunction[1]+beliefFunction[3] > this->m_PlausibilityThreshold)
      {
        out.Set(m_PowerSetIndexToLabelMap[1]); 
        precision += (1.0-beliefFunction[2]) - beliefFunction[1]; 
      }
      else
      {
        if (rand() > RAND_MAX/2)
          out.Set(m_PowerSetIndexToLabelMap[1]); 
        else
          out.Set(m_PowerSetIndexToLabelMap[2]); 
      }
    }
    else if (this->m_CombinationMode == 1)
    {
      if (beliefFunction[2] > beliefFunction[1] && beliefFunction[2] > beliefFunction[0])
      {
        out.Set(m_PowerSetIndexToLabelMap[2]); 
      }
      else if (beliefFunction[1] > beliefFunction[2] && beliefFunction[1] > beliefFunction[0])
      {
        out.Set(m_PowerSetIndexToLabelMap[1]); 
      }
      else
      {
        if (rand() > RAND_MAX/2)
          out.Set(m_PowerSetIndexToLabelMap[1]); 
        else
          out.Set(m_PowerSetIndexToLabelMap[2]); 
      }
    }
    
    believeIterator.Set(beliefFunction[2]); 
    plausibilityIterator.Set(beliefFunction[2]+beliefFunction[3]); 
    //std::cout << "believe function=" << beliefFunction[1] << "," << beliefFunction[2] << "," << beliefFunction[3] << std::endl; 
    totalConflict += conflict; 
    
    conflictIterator.Set(conflict); 
  }
  
  std::cout << "precision=" << precision << ", conflict=" << totalConflict << std::endl; 
  
  delete[] it;  
  
}


template< typename TInputImage, typename TOutputImage >
void
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::CalculateReliability(const OutputImageRegionType& outputRegionForThread)
{
  typedef itk::SegmentationReliabilityCalculator<TInputImage, TInputImage, TInputImage> SegmentationReliabilityFilterType;
  typedef itk::CastImageFilter<TInputImage, TInputImage> CastImageFilterType; 

  typename SegmentationReliabilityFilterType::Pointer reliabilityFilter = SegmentationReliabilityFilterType::New();
  typename CastImageFilterType::Pointer castFilter = CastImageFilterType::New(); 
  
  const unsigned int numberOfInputFiles = this->GetNumberOfInputs();
  for (unsigned int i = 0; i < numberOfInputFiles; i++)
  {
    reliabilityFilter->SetBaselineImage(this->GetOutput());
    reliabilityFilter->SetBaselineMask(this->GetOutput());
    castFilter->SetInput(this->GetInput(i)); 
    castFilter->Update(); 
    // TODO: slightly dodgy here - fix it later. 
    reliabilityFilter->SetRepeatImage(const_cast<TInputImage*>(this->GetInput(i))); 
    reliabilityFilter->SetRepeatMask(const_cast<TInputImage*>(this->GetInput(i)));
    reliabilityFilter->SetNumberOfErosion(1);
    reliabilityFilter->SetNumberOfDilation(1);
    reliabilityFilter->Compute();
    this->m_SegmentationReliability[i] = reliabilityFilter->GetSegmentationReliability(); 
    std::cout << reliabilityFilter->GetSegmentationReliability() << std::endl;
  }
  std::cout << std::endl;
  
}
  

template< typename TInputImage, typename TOutputImage >
void
DSSegmentationFusionImageFilter< TInputImage, TOutputImage >
::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, int itkNotUsed(threadId))
{
  
  CalculateExpectedSegmentation(outputRegionForThread); 
#if 0
  for (int i = 0; i < 3; i++)
  {
    CalculateExpectedSegmentation(outputRegionForThread); 
    CalculateReliability(outputRegionForThread); 
  } 
#endif
  
}

}

#endif 


