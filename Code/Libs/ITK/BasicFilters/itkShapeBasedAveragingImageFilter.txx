/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-06-04 10:01:00 +0100 (Fri, 04 Jun 2010) $
 Revision          : $Revision: 3342 $
 Last modified by  : $Author: kkl $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#ifndef ITKSHAPEBASEDAVERAGINGIMAGEFILTER_TXX_
#define ITKSHAPEBASEDAVERAGINGIMAGEFILTER_TXX_

#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include <map>
#include "../../../Prototype/kkl/STAPLE/itkSegmentationReliabilityCalculator.h"

namespace itk 
{


template<class TInputImage, class TOutputImage>
void 
ShapeBasedAveragingImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  typedef Image<int, TInputImage::ImageDimension> IntImageType; 
  typedef typename Superclass::DataObjectPointerArraySizeType ArraySizeType;
  typedef SignedMaurerDistanceMapImageFilter<IntImageType, FloatImageType> SignedMaurerDistanceMapImageFilterType;
  typedef CastImageFilter<TInputImage, IntImageType> CastImageFilterType; 
  typedef ImageRegionIteratorWithIndex<AverageDistanceMapType> AverageDistanceMapIteratorType; 
  typedef ImageRegionIterator<TOutputImage> OutputImageIteratorType; 
  typedef ImageRegionConstIterator<TInputImage> InputImageIteratorType; 
  typedef std::map<typename TInputImage::PixelType, int> LabelMapType; 
  unsigned short numberOfLabels = 0; 
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  std::cout << "Mean mode:" << this->m_MeanMode << std::endl; 
  // Forcing mean mode to simple mean if the number of input is less than 4. 
  if (numberOfInputs < 4 && m_MeanMode == INTERQUARTILE_MEAN)
  {
    m_MeanMode = MEAN; 
    std::cerr << "Warning: forcing to simple mean mode when the number of input is less than 4." << std::endl; 
  }
  
  this->m_SegmentationReliability.resize(numberOfInputs, 1.0); 
  
  // Look for different labels in the input images. 
  LabelMapType labelMap; 
  for (ArraySizeType imageIndex = 0; imageIndex < numberOfInputs; imageIndex++)
  {
    InputImageIteratorType it(this->GetInput(imageIndex), this->GetInput(imageIndex)->GetLargestPossibleRegion()); 
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      labelMap[it.Get()] = 1; 
    }
  }
  if (labelMap.size() >= std::numeric_limits<unsigned short>::max())
    itkExceptionMacro("ShapeBasedAveragingImageFilter: number of labels found greater than " << std::numeric_limits<unsigned short>::max() << ". Are you sure your input images are correct?");
  numberOfLabels = labelMap.size(); 
  if (!this->m_IsUserDefinedLabelForUndecidedPixels)
  {
    this->m_LabelForUndecidedPixels = static_cast<typename TOutputImage::PixelType>(labelMap.rbegin()->first+1); 
  }
  // Allocate space for the output merged image. 
  this->SetNumberOfOutputs(1); 
  this->AllocateOutputs(); 
  
  // Allocate space for the average distance map and initialise it to max.   
  typename FloatImageType::Pointer averageDistanceMap = FloatImageType::New();
  averageDistanceMap->SetOrigin(this->GetInput(0)->GetOrigin()); 
  averageDistanceMap->SetSpacing(this->GetInput(0)->GetSpacing()); 
  averageDistanceMap->SetRegions(this->GetInput(0)->GetLargestPossibleRegion()); 
  averageDistanceMap->Allocate(); 
  AverageDistanceMapIteratorType it(averageDistanceMap, averageDistanceMap->GetLargestPossibleRegion()); 
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    it.Set(std::numeric_limits<float>::max()); 
  }

  typename SignedMaurerDistanceMapImageFilterType::Pointer *distanceMapFilter = new typename SignedMaurerDistanceMapImageFilterType::Pointer[numberOfInputs];
  typename CastImageFilterType::Pointer *castImageFilter = new typename CastImageFilterType::Pointer[numberOfInputs]; 

  // Loop over all labels. 
  for (typename LabelMapType::iterator labelMapIt = labelMap.begin(); labelMapIt != labelMap.end(); ++labelMapIt)
  {
    typename TInputImage::PixelType label = labelMapIt->first;
        
    // Loop over all input images to calculate the distance transform.  
    for (ArraySizeType imageIndex = 0; imageIndex < numberOfInputs; imageIndex++)
    {
      castImageFilter[imageIndex] = CastImageFilterType::New();
      castImageFilter[imageIndex]->SetInput(this->GetInput(imageIndex)); 
      
      distanceMapFilter[imageIndex] = SignedMaurerDistanceMapImageFilterType::New(); 
      distanceMapFilter[imageIndex]->SetInput(castImageFilter[imageIndex]->GetOutput()); 
      distanceMapFilter[imageIndex]->SetUseImageSpacing(true); 
      
      // Filter only allows background value for the distance transform
      // therefore - have to set the background to the label and set the sign to be +ve inside
      // so that this keeps the convention the same as the paper (inside -ve, outside +ve). 
      distanceMapFilter[imageIndex]->SetBackgroundValue(label); 
      distanceMapFilter[imageIndex]->SetInsideIsPositive(true); 
      distanceMapFilter[imageIndex]->Update(); 
    }
      
    // Loop over all voxels. 
    AverageDistanceMapIteratorType averageDistanceMapIt(averageDistanceMap, averageDistanceMap->GetLargestPossibleRegion()); 
    OutputImageIteratorType outputImageIt(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion()); 
    
    for (averageDistanceMapIt.GoToBegin(), outputImageIt.GoToBegin();
        !averageDistanceMapIt.IsAtEnd();
        ++averageDistanceMapIt, ++outputImageIt)
    {
      // Compuate the average distance. 
      double averageDistance = 0; 
      std::vector<double> allDistances; 
      for (ArraySizeType imageIndex = 0; imageIndex < numberOfInputs; imageIndex++)
      {
        averageDistance += distanceMapFilter[imageIndex]->GetOutput()->GetPixel(averageDistanceMapIt.GetIndex()); 
        //allDistances.push_back(distanceMapFilter[imageIndex]->GetOutput()->GetPixel(averageDistanceMapIt.GetIndex())/this->m_SegmentationReliability[imageIndex]);
        allDistances.push_back(distanceMapFilter[imageIndex]->GetOutput()->GetPixel(averageDistanceMapIt.GetIndex()));  
      }
      switch (m_MeanMode)
      {
        case MEAN:
          averageDistance /= static_cast<double>(numberOfInputs); 
          break; 
          
        case MEDIAN: 
          sort(allDistances.begin(), allDistances.end()); 
          nth_element(allDistances.begin(), allDistances.begin()+allDistances.size()/2, allDistances.end());
          if ((allDistances.size() % 2) == 0)
            averageDistance = (*(allDistances.begin()+allDistances.size()/2) + *(allDistances.begin()+allDistances.size()/2-1))/2;
          else
            averageDistance = *(allDistances.begin()+allDistances.size()/2);
          break; 
          
        case INTERQUARTILE_MEAN: 
        {
          int start = static_cast<int>(floor(static_cast<double>(numberOfInputs)/4.0)); 
          int end = static_cast<int>(floor(3.0*static_cast<double>(numberOfInputs)/4.0))-1; 
          
          sort(allDistances.begin(), allDistances.end()); 
          averageDistance = 0.0; 
          for (int i = start; i <= end; i++)
            averageDistance = *(allDistances.begin()+i); 
          averageDistance /= static_cast<double>(end-start+1); 
          break; 
        }
          
        default: 
          assert(false); 
      }
      // Update the distance map and output image. 
      if (averageDistance < averageDistanceMapIt.Get())
      {
        outputImageIt.Set(static_cast<typename TOutputImage::PixelType>(label)); 
        averageDistanceMapIt.Set(static_cast<float>(averageDistance)); 
      }
      else if (averageDistance == averageDistanceMapIt.Get())
      {
        // Quick hack to have some randomness when the voxels are equi-distance from two labels. 
        if (this->m_LabelForUndecidedPixels == 240)
        {
          if (static_cast<double>(rand())/static_cast<double>(RAND_MAX) < 0.5)
            outputImageIt.Set(static_cast<typename TOutputImage::PixelType>(label)); 
        }
        else
        {
          outputImageIt.Set(this->m_LabelForUndecidedPixels); 
        }
      }
    }
        
    //std::cout << "totalVariance=" << CalculateVariance(averageDistanceMap) << std::endl; 
  }
  if (distanceMapFilter != NULL) 
    delete [] distanceMapFilter;
  if (castImageFilter != NULL) 
    delete [] castImageFilter;
  
}



template<class TInputImage, class TOutputImage>
void 
ShapeBasedAveragingImageFilter<TInputImage, TOutputImage>
::CalculateReliability()
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


template<class TInputImage, class TOutputImage>
double 
ShapeBasedAveragingImageFilter<TInputImage, TOutputImage>
::CalculateVariance(typename AverageDistanceMapType::Pointer averageDistanceMap)
{
  typedef Image<int, TInputImage::ImageDimension> IntImageType; 
  typedef typename Superclass::DataObjectPointerArraySizeType ArraySizeType;
  typedef SignedMaurerDistanceMapImageFilter<IntImageType, FloatImageType> SignedMaurerDistanceMapImageFilterType;
  typedef CastImageFilter<TInputImage, IntImageType> CastImageFilterType; 
  typedef ImageRegionIteratorWithIndex<AverageDistanceMapType> AverageDistanceMapIteratorType; 
  typedef ImageRegionConstIterator<TInputImage> InputImageIteratorType; 
  typedef std::map<typename TInputImage::PixelType, int> LabelMapType; 
  
  double totalVariance = 0.0; 
  
  // Look for different labels in the input images. 
  LabelMapType labelMap; 
  for (ArraySizeType imageIndex = 0; imageIndex < this->GetNumberOfInputs(); imageIndex++)
  {
    InputImageIteratorType it(this->GetInput(imageIndex), this->GetInput(imageIndex)->GetLargestPossibleRegion()); 
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      labelMap[it.Get()] = 1; 
    }
  }
  
  typename SignedMaurerDistanceMapImageFilterType::Pointer distanceMapFilter[this->GetNumberOfInputs()]; 
  typename CastImageFilterType::Pointer castImageFilter[this->GetNumberOfInputs()]; 
  
  // Loop over all labels. 
  for (typename LabelMapType::iterator labelMapIt = labelMap.begin(); labelMapIt != labelMap.end(); ++labelMapIt)
  {
    typename TInputImage::PixelType label = labelMapIt->first;
        
    // Loop over all input images to calculate the distance transform.  
    for (ArraySizeType imageIndex = 0; imageIndex < this->GetNumberOfInputs(); imageIndex++)
    {
      castImageFilter[imageIndex] = CastImageFilterType::New();
      castImageFilter[imageIndex]->SetInput(this->GetInput(imageIndex)); 
      distanceMapFilter[imageIndex] = SignedMaurerDistanceMapImageFilterType::New(); 
      distanceMapFilter[imageIndex]->SetInput(castImageFilter[imageIndex]->GetOutput()); 
      distanceMapFilter[imageIndex]->SetUseImageSpacing(true); 
      // Filter only allows background value for the distance transform
      // therefore - have to set the background to the label and set the sign to be +ve inside
      // so that this keeps the convention the same as the paper (inside -ve, outside +ve). 
      distanceMapFilter[imageIndex]->SetBackgroundValue(label); 
      distanceMapFilter[imageIndex]->SetInsideIsPositive(true); 
      distanceMapFilter[imageIndex]->Update(); 
    }
      
    // Loop over all voxels. 
    AverageDistanceMapIteratorType averageDistanceMapIt(averageDistanceMap, averageDistanceMap->GetLargestPossibleRegion()); 
    
    for (averageDistanceMapIt.GoToBegin();
        !averageDistanceMapIt.IsAtEnd();
        ++averageDistanceMapIt)
    {
      // Compuate the average distance. 
      double averageDistance = averageDistanceMapIt.Get(); 
      std::vector<double> allDistances; 
      for (ArraySizeType imageIndex = 0; imageIndex < this->GetNumberOfInputs(); imageIndex++)
      {
        allDistances.push_back(distanceMapFilter[imageIndex]->GetOutput()->GetPixel(averageDistanceMapIt.GetIndex()));  
      }
        
      double currentVariance = 0.0; 
      for (unsigned int j = 0; j < allDistances.size(); j++)
        currentVariance += (averageDistance-allDistances[j])*(averageDistance-allDistances[j]); 
      currentVariance /= allDistances.size()-1; 
      totalVariance += currentVariance; 
    }
  }
  
  return totalVariance; 
}


}

#endif // ITKSHAPEBASEDAVERAGINGIMAGEFILTER_TXX_



