/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSetOutputVectorToCurrentPositionFilter_txx
#define __itkSetOutputVectorToCurrentPositionFilter_txx

#include "itkSetOutputVectorToCurrentPositionFilter.h"
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
SetOutputVectorToCurrentPositionFilter<TScalarType, NDimensions>
::SetOutputVectorToCurrentPositionFilter()
{
  m_OutputIsInMillimetres = true;
  
  niftkitkDebugMacro(<<"SetOutputVectorToCurrentPositionFilter():Constructed with m_OutputIsInMillimetres=" << m_OutputIsInMillimetres \
      );
}

template <class TScalarType, unsigned int NDimensions>
void
SetOutputVectorToCurrentPositionFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "OutputIsInMillimetres = " << m_OutputIsInMillimetres << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void
SetOutputVectorToCurrentPositionFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 1 inputs.
  if (numberOfInputs != 1)
    {
      itkExceptionMacro(<< "SetOutputVectorToCurrentPositionFilter should only have 1 input.");
    }
}

template <class TScalarType, unsigned int NDimensions>
void
SetOutputVectorToCurrentPositionFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, ThreadIdType threadNumber)
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  // Get Pointers to images.
  typename InputImageType::Pointer inputImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionIteratorWithIndex<InputImageType> inputIterator(inputImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputRegionForThread);
  
  InputImageIndexType  index;
  OutputImagePointType point;
  OutputPixelType      pixel;
  
  for (inputIterator.GoToBegin(), 
       outputIterator.GoToBegin(); 
       !inputIterator.IsAtEnd(); 
       ++inputIterator, 
       ++outputIterator)
    {
      index = inputIterator.GetIndex();
      
      if (m_OutputIsInMillimetres)
        {
          inputImage->TransformIndexToPhysicalPoint( index, point );
          
          for (unsigned int i=0; i < Dimension; i++)
            {
              pixel[i] = point[i];  
            }
          
        }
      else
        {
          for (unsigned int i=0; i < Dimension; i++)
            {
              pixel[i] = index[i];  
            }
        }
      outputIterator.Set(pixel);
    }

  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


} // end namespace

#endif
