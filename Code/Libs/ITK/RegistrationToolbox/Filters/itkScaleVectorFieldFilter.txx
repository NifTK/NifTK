/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkScaleVectorFieldFilter_txx
#define __itkScaleVectorFieldFilter_txx

#include "itkScaleVectorFieldFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
ScaleVectorFieldFilter<TScalarType, NDimensions>
::ScaleVectorFieldFilter()
{
}

template <class TScalarType, unsigned int NDimensions>
void
ScaleVectorFieldFilter<TScalarType, NDimensions>
::SetNthInput(unsigned int idx, const InputImageType *image)
{
  this->ProcessObject::SetNthInput(idx, const_cast< InputImageType* >(image));
  this->Modified();
  
  niftkitkDebugMacro(<<"SetNthInput():Set input[" << idx << "] to address:" << image);
}

template <class TScalarType, unsigned int NDimensions>
void
ScaleVectorFieldFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template <class TScalarType, unsigned int NDimensions>
void
ScaleVectorFieldFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 2 inputs.
  if (numberOfInputs != 2)
    {
      itkExceptionMacro(<< "ScaleVectorFieldFilter should only have 2 inputs.");
    }

  InputImageRegionType region;
  for (unsigned int i=0; i<numberOfInputs; i++)
    {
      // Check each input is set.
      InputImageType *input = static_cast< InputImageType * >(this->ProcessObject::GetInput(i));
      if (!input)
        {
          itkExceptionMacro(<< "Input " << i << " not set!");
        }
        
      // Check they are the same size.
      if (i==0)
        {
          region = input->GetLargestPossibleRegion();
        }
      else if (input->GetLargestPossibleRegion() != region) 
        {
          itkExceptionMacro(<< "All Inputs must have the same dimensions.");
        }
    }
}

template <class TScalarType, unsigned int NDimensions>
void
ScaleVectorFieldFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  // Get Pointers to images.
  typename InputImageType::ConstPointer imageThatGetsScaled 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename InputImageType::ConstPointer imageThatDoesTheScaling 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
    
  // Make iterators of the right region size.

  typedef ImageRegionIterator< OutputImageType > OutputImageIteratorType;
  OutputImageIteratorType outputIterator(outputImage, outputRegionForThread);  

  typedef ImageRegionConstIterator < InputImageType > InputImageIteratorType;
  InputImageIteratorType imageThatGetsScaledIterator(imageThatGetsScaled, outputRegionForThread);
  InputImageIteratorType imageThatDoesTheScalingIterator(imageThatDoesTheScaling, outputRegionForThread);
  
  outputIterator.GoToBegin();
  imageThatGetsScaledIterator.GoToBegin();
  imageThatDoesTheScalingIterator.GoToBegin();
  
  InputImagePixelType imageThatGetsScaledValue;
  InputImagePixelType imageThatDoesTheScalingValue;
  OutputPixelType     outputValue;
  TScalarType         magnitude;
  unsigned int        i;
  
  while(!outputIterator.IsAtEnd() && !imageThatGetsScaledIterator.IsAtEnd() && !imageThatDoesTheScalingIterator.IsAtEnd())
    {
    
      imageThatGetsScaledValue = imageThatGetsScaledIterator.Get();
      imageThatDoesTheScalingValue = imageThatDoesTheScalingIterator.Get();
      
      if (m_ScaleByComponents)
        {
          for (i = 0; i < NDimensions; i++)
            {
              outputValue[i] = imageThatGetsScaledValue[i] * fabs(imageThatDoesTheScalingValue[i]);    
            }
        }
      else
        {
          magnitude = 0;
          
          for (i = 0; i < NDimensions; i++)
            {
              magnitude += (imageThatDoesTheScalingValue[i] * imageThatDoesTheScalingValue[i]);
            }
          
          magnitude = sqrt(magnitude);
          
          for (i = 0; i < NDimensions; i++)
            {
              outputValue[i] = imageThatGetsScaledValue[i] * magnitude;
            }
        }
      
      outputIterator.Set(outputValue);
      
      ++outputIterator;
      ++imageThatGetsScaledIterator;
      ++imageThatDoesTheScalingIterator;
    }
    
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}

template <class TScalarType, unsigned int NDimensions>
void
ScaleVectorFieldFilter<TScalarType, NDimensions>
::WriteVectorImage(std::string filename)
{
  niftkitkDebugMacro(<<"WriteVectorImage():Writing to:" << filename);
  
  typedef float OutputVectorDataType;
  typedef Vector<OutputVectorDataType, NDimensions> OutputVectorPixelType;
  typedef Image<OutputVectorPixelType, NDimensions> OutputVectorImageType;
  typedef CastImageFilter<OutputImageType, OutputVectorImageType> CastFilterType;
  typedef ImageFileWriter<OutputVectorImageType> WriterType;
  
  typename CastFilterType::Pointer caster = CastFilterType::New();
  typename WriterType::Pointer writer = WriterType::New();

  caster->SetInput(this->GetOutput());
  writer->SetFileName(filename);
  writer->SetInput(caster->GetOutput());
  writer->Update();
  
  niftkitkDebugMacro(<<"WriteVectorImage():Writing to:" << filename << "....DONE");
}

} // end namespace itk

#endif
