/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVectorVPlusLambdaUImageFilter_txx
#define __itkVectorVPlusLambdaUImageFilter_txx

#include "itkVectorVPlusLambdaUImageFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkUCLMacro.h>
#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
VectorVPlusLambdaUImageFilter<TScalarType, NDimensions>
::VectorVPlusLambdaUImageFilter()
{
  m_Lambda = 1;
  m_IgnoreInputV = false;
  m_SubtractSteps = false;
  
  niftkitkDebugMacro(<< "VectorVPlusLambdaUImageFilter():Constructed with m_Lambda=" << m_Lambda \
      << ", m_IgnoreInputV=" << m_IgnoreInputV \
      << ", m_SubtractSteps=" << m_SubtractSteps \
      );  
}

template <class TScalarType, unsigned int NDimensions>
void
VectorVPlusLambdaUImageFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Lambda = " << m_Lambda << std::endl;
  os << indent << "IgnoreInputV = " << m_IgnoreInputV << std::endl;
  os << indent << "SubtractSteps = " << m_SubtractSteps << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void
VectorVPlusLambdaUImageFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 2 inputs.
  if (numberOfInputs != 2)
    {
      niftkitkExceptionMacro(<< "VectorVPlusLambdaUImageFilter should have 2 inputs.");
    }
  
  InputImageRegionType region;
  for (unsigned int i=0; i<numberOfInputs; i++)
    {
      // Check each input is set.
      InputImageType *input = static_cast< InputImageType * >(this->ProcessObject::GetInput(i));
      if (!input)
        {
          niftkitkExceptionMacro(<< "Input " << i << " not set!");
        }
        
      // Check they are the same size.
      if (i==0)
        {
          region = input->GetLargestPossibleRegion();
        }
      else if (input->GetLargestPossibleRegion() != region) 
        {
          niftkitkExceptionMacro(<< "All Inputs must have the same dimensions.");
        }
    }  
}

template <class TScalarType, unsigned int NDimensions>
void
VectorVPlusLambdaUImageFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, ThreadIdType threadNumber)
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  // Get Pointers to images.
  typename InputImageType::Pointer vImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename InputImageType::Pointer uImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionConstIterator<InputImageType> vIterator(vImage, outputRegionForThread);
  ImageRegionConstIterator<InputImageType> uIterator(uImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputRegionForThread);
  
  InputPixelType u;
  InputPixelType v;
  InputPixelType output;
  unsigned int i = 0;
  
  double factor = 1;
  
  if (m_SubtractSteps)
    {
      factor = -1;
    }
    
  for (vIterator.GoToBegin(),
       uIterator.GoToBegin(),
       outputIterator.GoToBegin(); 
       !vIterator.IsAtEnd(); 
       ++vIterator,
       ++uIterator,
       ++outputIterator)
    {
      v = vIterator.Get();
      u = uIterator.Get();
      
      for (i = 0; i < Dimension; i++)
        {
          if (m_IgnoreInputV)
            {
              output[i] = factor * m_Lambda * u[i];
            }
          else
            {
              output[i] = v[i] + factor * m_Lambda * u[i];    
            }
        }
      outputIterator.Set(output);
    }

  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


} // end namespace

#endif
