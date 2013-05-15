/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDasTransformImageFilter_txx
#define __itkDasTransformImageFilter_txx

#include "itkDasTransformImageFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
DasTransformImageFilter<TScalarType, NDimensions>
::DasTransformImageFilter()
{
  m_FileName = "tmp.tmi.nii";
  m_WriteTransformedMovingImage = false;
  m_DefaultValue = 0;
  
  m_Interpolator = LinearInterpolatorType::New();
  
  niftkitkDebugMacro(<<"DasGradientFilter():Constructed with m_FileName=" << m_FileName \
      << ", m_WriteTransformedMovingImage=" << m_WriteTransformedMovingImage \
      << ", m_DefaultValue=" << m_DefaultValue \
      );
}

template <class TScalarType, unsigned int NDimensions>
void
DasTransformImageFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "FileName = " << m_FileName << std::endl;
  os << indent << "WriteTransformedMovingImage = " << m_WriteTransformedMovingImage << std::endl;
  os << indent << "m_DefaultValue = " << m_DefaultValue << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void
DasTransformImageFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Started");
  
  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 1 inputs, set using the SetInput method.
  if (numberOfInputs != 1)
    {
      itkExceptionMacro(<< "DasTransformImageFilter should have 1 input set using SetInput(0, *image)");
    }
  
  // These are simply set as pointers, so managed externally to this class.
  if (m_PhiTransformation.IsNull())
    {
      itkExceptionMacro(<< "DasTransformImageFilter: m_PhiTransformation is null");  
    }
  
  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Finished");
}

template <class TScalarType, unsigned int NDimensions>
void
DasTransformImageFilter<TScalarType, NDimensions>
::AfterThreadedGenerateData( void )
{
  niftkitkDebugMacro(<<"AfterThreadedGenerateData():Started");
  
  if (m_WriteTransformedMovingImage)
    {
      typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
      
      typedef ImageFileWriter<OutputImageType> WriterType;
      typename WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(m_FileName);
      writer->SetInput(outputImage);
      writer->Update();
    }
  else
    {
	  niftkitkDebugMacro(<<"AfterThreadedGenerateData():Nothing to do");
    }
  
  niftkitkDebugMacro(<<"AfterThreadedGenerateData():Finished");
}

template <class TScalarType, unsigned int NDimensions>
void
DasTransformImageFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
{
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  // Input 0 is the image being transformed
  typename InputImageType::Pointer inputImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  // The output image is actually the same type (see header file)
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  // Make sure memory is there.
  this->AllocateOutputs();
  
  ImageRegionConstIterator<VectorImageType> phiIterator(m_PhiTransformation, outputRegionForThread);
  ImageRegionIterator<InputImageType> outputIterator(outputImage, outputRegionForThread);

  m_Interpolator->SetInputImage(inputImage);
  
  OutputPixelType outputPixel;
  VectorPixelType phiPixel;
  Point<TScalarType, Dimension> phiPoint;
  ContinuousIndex<TScalarType, Dimension> continousIndex; 
  
  for (phiIterator.GoToBegin(),
       outputIterator.GoToBegin();
       !phiIterator.IsAtEnd();
       ++phiIterator,
       ++outputIterator)
    {
      phiPixel = phiIterator.Get();
      
      for (unsigned int j = 0; j < Dimension; j++)
        {
          phiPoint[j] = phiPixel[j]; 
        }

      if (inputImage->TransformPhysicalPointToContinuousIndex(phiPoint, continousIndex))
        {
          outputPixel = m_Interpolator->Evaluate(phiPoint);
          outputIterator.Set(outputPixel);
        }           
      else
        {
          outputIterator.Set(m_DefaultValue);  
        }
    }
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}

} // end namespace

#endif
