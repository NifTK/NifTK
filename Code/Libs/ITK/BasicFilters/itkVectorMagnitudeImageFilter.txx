/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVectorMagnitudeImageFilter_txx
#define __itkVectorMagnitudeImageFilter_txx

#include "itkVectorMagnitudeImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"


namespace itk {

template <class TScalarType, unsigned int NDimensions>
VectorMagnitudeImageFilter<TScalarType, NDimensions>
::VectorMagnitudeImageFilter()
{
}

template <class TScalarType, unsigned int NDimensions>
void
VectorMagnitudeImageFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template <class TScalarType, unsigned int NDimensions>
void
VectorMagnitudeImageFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 1 inputs.
  if (numberOfInputs != 1)
    {
      itkExceptionMacro(<< "VectorMagnitudeImageFilter should only have 1 input.");
    }
}

template <class TScalarType, unsigned int NDimensions>
void
VectorMagnitudeImageFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  // Get Pointers to images.
  typename InputImageType::Pointer inputImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionConstIterator<InputImageType> inputIterator(inputImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputRegionForThread);
  
  double          squaredMagnitude;
  
  for (inputIterator.GoToBegin(), 
       outputIterator.GoToBegin(); 
       !inputIterator.IsAtEnd(); 
       ++inputIterator, 
       ++outputIterator)
    {
      
      squaredMagnitude = inputIterator.Get().GetSquaredNorm();
      outputIterator.Set(vcl_sqrt(squaredMagnitude));
    }

  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


} // end namespace

#endif
