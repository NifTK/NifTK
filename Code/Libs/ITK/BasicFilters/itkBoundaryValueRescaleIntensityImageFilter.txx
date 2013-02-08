/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBoundaryValueRescaleIntensityImageFilter_txx
#define __itkBoundaryValueRescaleIntensityImageFilter_txx

#include "itkBoundaryValueRescaleIntensityImageFilter.h"

#include "itkLogHelper.h"

namespace itk
{

template <typename TImageType>
BoundaryValueRescaleIntensityImageFilter<TImageType>
::BoundaryValueRescaleIntensityImageFilter()
{
  // Initialise these parameters to reasonable values.
  m_Scale = std::numeric_limits<RealType>::min();
  m_Shift = std::numeric_limits<RealType>::min();
  m_InputLowerThreshold = std::numeric_limits<PixelType>::min();
  m_InputUpperThreshold = std::numeric_limits<PixelType>::max();
  m_InputMinimum = std::numeric_limits<PixelType>::max();
  m_InputMaximum = std::numeric_limits<PixelType>::min();
  m_OutputMinimum = std::numeric_limits<PixelType>::min();
  m_OutputMaximum = std::numeric_limits<PixelType>::max();
  m_OutputBoundaryValue = std::numeric_limits<PixelType>::min();
  m_ThresholdFilter = BinaryThresholdFilterType::New();  
}

template <class TImageType>
void 
BoundaryValueRescaleIntensityImageFilter<TImageType>
::BeforeThreadedGenerateData()
{
  ImageType *input = static_cast< ImageType * >(this->ProcessObject::GetInput(0));
  
  m_ThresholdFilter->SetInput(input);
  m_ThresholdFilter->SetInsideValue(1);
  m_ThresholdFilter->SetOutsideValue(0);

  // Basically, threshold the image.

  if(m_InputLowerThreshold != std::numeric_limits<PixelType>::min())
    {
      niftkitkDebugMacro(<<"BeforeThreadedGenerateData():User has set lower threshold to:" << m_ThresholdFilter->GetLowerThreshold());
      m_ThresholdFilter->SetLowerThreshold(m_InputLowerThreshold);    
    }
  else
    {
      niftkitkDebugMacro(<<"BeforeThreadedGenerateData():User has not set a lower threshold");
      m_ThresholdFilter->SetLowerThreshold(std::numeric_limits<PixelType>::min());
    }
  
  if(m_InputUpperThreshold != std::numeric_limits<PixelType>::max())
    {
      niftkitkDebugMacro(<<"BeforeThreadedGenerateData():User has set upper threshold to:" << m_ThresholdFilter->GetUpperThreshold());
      m_ThresholdFilter->SetUpperThreshold(m_InputUpperThreshold);    
    }
  else
    {
      niftkitkDebugMacro(<<"BeforeThreadedGenerateData():User has not set an upper threshold");
      m_ThresholdFilter->SetUpperThreshold(std::numeric_limits<PixelType>::max());
    }
    
  m_ThresholdFilter->UpdateLargestPossibleRegion();
  
  // Then, for WHATS LEFT (i.e. stuff that isn't thresholded), we calculate min and max.

  ImageRegionConstIterator<ImageType> thresholdedIterator = ImageRegionConstIterator<ImageType>(m_ThresholdFilter->GetOutput(), m_ThresholdFilter->GetOutput()->GetLargestPossibleRegion());
  ImageRegionConstIterator<ImageType> inputIterator = ImageRegionConstIterator<ImageType>(input, input->GetLargestPossibleRegion());
  
  m_InputMinimum = std::numeric_limits<PixelType>::max();
  m_InputMaximum = std::numeric_limits<PixelType>::min();
  
  PixelType val;
  
  for (thresholdedIterator.GoToBegin(), inputIterator.GoToBegin();
       !thresholdedIterator.IsAtEnd() && !inputIterator.IsAtEnd();
       ++thresholdedIterator, ++inputIterator)
    {
      if (thresholdedIterator.Get() == 1)
        {
          val = inputIterator.Get(); 
          if (val > m_InputMaximum)
            {
              m_InputMaximum = val;
            }
          if (val < m_InputMinimum)
            {
              m_InputMinimum = val;
            }
        }
    }
  
  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Max is:" << m_InputMaximum << ", min is:" << m_InputMinimum);
  
  // Now calculate the rescaled range.
  if (m_InputMinimum != m_InputMaximum && m_OutputMinimum < m_OutputMaximum)
    {
      m_Scale = 
        (static_cast<RealType>( m_OutputMaximum )
         - static_cast<RealType>( m_OutputMinimum )) /
        (static_cast<RealType>( m_InputMaximum )
         - static_cast<RealType>( m_InputMinimum ));      
    }
  else
    {
      m_Scale = 0;
    }
  
  m_Shift =
    static_cast<RealType>( m_OutputMinimum ) - 
    static_cast<RealType>( m_InputMinimum ) * m_Scale;

  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Shift is:" << m_Shift << ", scale is:" << m_Scale);
  
}

template <class TImageType>
void
BoundaryValueRescaleIntensityImageFilter<TImageType>
::ThreadedGenerateData(const ImageRegionType& outputRegionForThread, int threadNumber) 
{
  
  ImageType *thresholdedInput = m_ThresholdFilter->GetOutput();
  ImageType *imageInput = static_cast< ImageType * >(this->ProcessObject::GetInput(0));
  ImageType *outputImage = static_cast< ImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionConstIterator<ImageType> thresholdedIterator = ImageRegionConstIterator<ImageType>(thresholdedInput, outputRegionForThread);
  ImageRegionConstIterator<ImageType> imageIterator = ImageRegionConstIterator<ImageType>(imageInput, outputRegionForThread);
  ImageRegionIterator<ImageType> outputIterator = ImageRegionIterator<ImageType>(outputImage, outputRegionForThread);
  PixelType val;
  
  for (thresholdedIterator.GoToBegin(), imageIterator.GoToBegin(), outputIterator.GoToBegin();
       !thresholdedIterator.IsAtEnd() && !imageIterator.IsAtEnd() && !outputIterator.IsAtEnd();
       ++thresholdedIterator, ++imageIterator, ++outputIterator)
    {
      if (thresholdedIterator.Get() == 1)
        {          
          val = imageIterator.Get();

          RealType value  = static_cast<RealType>(val) * m_Scale + m_Shift;
          RealType  result = static_cast<RealType>( value );
          result = ( result > m_OutputMaximum ) ? m_OutputMaximum : result;
          result = ( result < m_OutputMinimum ) ? m_OutputMinimum : result;

          outputIterator.Set((PixelType)result);
        }
      else
        {
          outputIterator.Set(m_OutputBoundaryValue);  
        }
    }
    
}

template <class TImageType>
void 
BoundaryValueRescaleIntensityImageFilter<TImageType>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Scale: " << m_Scale << std::endl;
  os << indent << "Shift: " << m_Shift << std::endl;
  os << indent << "InputLowerThreshold: " << m_InputLowerThreshold << std::endl;
  os << indent << "InputUpperThreshold: " << m_InputUpperThreshold << std::endl;
  os << indent << "InputMinimum: " << m_InputMinimum << std::endl;
  os << indent << "InputMaximum: " << m_InputMaximum << std::endl;
  os << indent << "OutputMinimum: " << m_OutputMinimum << std::endl;
  os << indent << "OutputMaximum: " << m_OutputMaximum << std::endl;
  os << indent << "OutputBoundaryValue: " << m_OutputBoundaryValue << std::endl;  
}

} /** End namespace. */

#endif
