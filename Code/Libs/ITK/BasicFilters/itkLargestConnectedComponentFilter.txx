/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkLargestConnectedComponentFilter_txx
#define itkLargestConnectedComponentFilter_txx

#include <map>
#include <algorithm>

#include "itkLargestConnectedComponentFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"

using namespace std;

namespace itk
{
  //Constructor
  template <class TInputImage, class TOutputImage>
  LargestConnectedComponentFilter<TInputImage, TOutputImage>::LargestConnectedComponentFilter()
  {
    this->SetNumberOfRequiredOutputs(1);
    m_InputBackgroundValue = 0;
    m_OutputBackgroundValue = 0;
    m_OutputForegroundValue = 1;
    m_CastFilter = CastImageFilterType::New();
    m_ConnectedFilter = ConnectedComponentFilterType::New();
  }


  template <class TInputImage, class TOutputImage>
  void LargestConnectedComponentFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_InputBackgroundValue=" << m_InputBackgroundValue << std::endl;
    os << indent << "m_OutputBackgroundValue=" << m_OutputBackgroundValue << std::endl;
    os << indent << "m_OutputForegroundValue=" << m_OutputForegroundValue << std::endl;
  }


  template <class TInputImage, class TOutputImage>
  void LargestConnectedComponentFilter<TInputImage, TOutputImage>::GenerateData()
  { 
    this->AllocateOutputs();

    // Check only one input.    
    const unsigned int numberOfInputImages = this->GetNumberOfInputs();
    if(numberOfInputImages != 1)
    {
      itkExceptionMacro(<< "There should be one input image for LargestConnectedComponentFilter. ");
    }
    
    // Check input image is set.
    InputImageType *inputImage = static_cast<InputImageType*>(this->ProcessObject::GetInput(0));
    if(!inputImage)
    {
      itkExceptionMacro(<< "Input image is not set!");
    }
    
    typedef itk::ImageRegionConstIterator<InternalImageType> ImageRegionConstIteratorType;
    typedef itk::ImageRegionIterator<OutputImageType>        ImageRegionIteratorType;
    
    m_CastFilter->SetInput(this->GetInput());
    m_ConnectedFilter->SetInput(m_CastFilter->GetOutput());
    m_ConnectedFilter->SetBackgroundValue(m_InputBackgroundValue);
    m_ConnectedFilter->SetFullyConnected(false);
    m_ConnectedFilter->UpdateLargestPossibleRegion();
    
    // Count the number of voxels in each components.
    std::map<InternalPixelType, unsigned long int> componentSizes;
    InternalPixelType                              largestSizeLabel = 0; 
    unsigned long int                              largestSize = 0; 
    
    ImageRegionConstIteratorType ccIt(m_ConnectedFilter->GetOutput(), m_ConnectedFilter->GetOutput()->GetLargestPossibleRegion());
    for (ccIt.GoToBegin(); !ccIt.IsAtEnd(); ++ccIt)
    {
      if (ccIt.Get() != (int)m_InputBackgroundValue)
      {
        componentSizes[ccIt.Get()]++;
      }
         
    }
    for (std::map<InternalPixelType, unsigned long int>::iterator it = componentSizes.begin(); it != componentSizes.end(); it++)
    {
      if (it->second > largestSize)
      {
        largestSize = it->second; 
        largestSizeLabel = it->first; 
      }
    }

    if (largestSizeLabel == (int)m_InputBackgroundValue)
    {
      this->GetOutput()->FillBuffer(m_OutputBackgroundValue);
    }
    else
    {
      ImageRegionIteratorType outIt(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());
      for (ccIt.GoToBegin(), outIt.GoToBegin(); !ccIt.IsAtEnd(); ++ccIt, ++outIt)
      {
        if (ccIt.Get() == largestSizeLabel)
        {
          outIt.Set(m_OutputForegroundValue);
        }
        else
        {
          outIt.Set(m_OutputBackgroundValue);
        }
      }    
    }
  }//end of generatedata method

}//end namespace itk


#endif
