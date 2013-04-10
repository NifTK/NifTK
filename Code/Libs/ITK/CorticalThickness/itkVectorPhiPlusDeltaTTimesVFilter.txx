/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVectorPhiPlusDeltaTTimesVFilter_txx
#define __itkVectorPhiPlusDeltaTTimesVFilter_txx

#include "itkVectorPhiPlusDeltaTTimesVFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk {

template <class TScalarType, unsigned int NDimensions>
VectorPhiPlusDeltaTTimesVFilter<TScalarType, NDimensions>
::VectorPhiPlusDeltaTTimesVFilter()
{
  m_DeltaT = 1;
  m_NumberOfSteps = 1;
  m_SubtractSteps = false;
  
  m_Interpolator = VectorLinearInterpolatorType::New();
  m_PhiZeroTransformation = InputImageType::New();
  m_ThicknessPriorImage = InputScalarImageType::New();
  m_ThicknessImage = InputScalarImageType::New();
  
  niftkitkDebugMacro(<<"VectorVPlusLambdaUImageFilter():Constructed with m_DeltaT=" << m_DeltaT \
      << ", m_NumberOfSteps=" << m_NumberOfSteps \
      << ", m_SubtractSteps=" << m_SubtractSteps \
      );  
}

template <class TScalarType, unsigned int NDimensions>
void
VectorPhiPlusDeltaTTimesVFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "DeltaT = " << m_DeltaT << std::endl;
  os << indent << "NumberOfSteps = " << m_NumberOfSteps << std::endl;
  os << indent << "SubtractSteps = " << m_SubtractSteps << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void
VectorPhiPlusDeltaTTimesVFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  // Check to verify all inputs are specified and have the same metadata, spacing etc...
  
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 2 inputs, set using the SetInput method.
  if (numberOfInputs != 2)
    {
      itkExceptionMacro(<< "VectorPhiPlusDeltaTTimesVFilter should have 2 inputs set using SetInput([0,1], *image)");
    }
  
  // These are simply set as pointers, so managed externally to this class.
  if (m_PhiZeroTransformation.IsNull())
    {
      itkExceptionMacro(<< "VectorPhiPlusDeltaTTimesVFilter: m_PhiZeroTransformation is null");  
    }
  
  if (m_ThicknessPriorImage.IsNull())
    {
      itkExceptionMacro(<< "VectorPhiPlusDeltaTTimesVFilter: m_ThicknessPriorImage is null");  
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
  
  // Allocate the thickness image.
  InputImageType *input = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  m_ThicknessImage->SetRegions(input->GetLargestPossibleRegion());
  m_ThicknessImage->SetDirection(input->GetDirection());
  m_ThicknessImage->SetOrigin(input->GetOrigin());
  m_ThicknessImage->SetSpacing(input->GetSpacing());
  m_ThicknessImage->Allocate();
  m_ThicknessImage->FillBuffer(0);
}

template <class TScalarType, unsigned int NDimensions>
void
VectorPhiPlusDeltaTTimesVFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber << ", dt=" << m_DeltaT << ", steps=" << m_NumberOfSteps);

  Point<TScalarType, Dimension> phiPoint;
  ContinuousIndex<TScalarType, Dimension> continousIndex; 

  // Get Pointers to images.
  typename InputImageType::Pointer phiImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename InputImageType::Pointer velocityImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(1));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionConstIteratorWithIndex<InputImageType> phiIterator(phiImage, outputRegionForThread);
  ImageRegionConstIterator<InputImageType> phiZeroIterator(m_PhiZeroTransformation, outputRegionForThread);
  ImageRegionConstIterator<InputScalarImageType> thicknessPriorIterator(m_ThicknessPriorImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputRegionForThread);
  ImageRegionIterator<InputScalarImageType> thicknessIterator(m_ThicknessImage, outputRegionForThread);
  
  m_Interpolator->SetInputImage(velocityImage);
  
  InputPixelType phiPixel;
  InputPixelType phiZeroPixel;
  InputPixelType vPixel;
  InputPixelType outputPixel;
  TScalarType    thicknessPriorPixel;
  
  unsigned int i = 0;
  unsigned int j = 0;
  double thickness = 0;
  double factor = 1;
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Running with m_SubtractSteps=" << m_SubtractSteps);
  if (m_SubtractSteps)
    {
      factor = -1;
    }

  for (phiIterator.GoToBegin(),
       phiZeroIterator.GoToBegin(),
       thicknessPriorIterator.GoToBegin(),
       thicknessIterator.GoToBegin(),
       outputIterator.GoToBegin(); 
       !phiIterator.IsAtEnd(); 
       ++phiIterator,
       ++phiZeroIterator,
       ++thicknessPriorIterator,
       ++thicknessIterator,
       ++outputIterator)
    {
      phiPixel = phiIterator.Get();
      phiZeroPixel = phiZeroIterator.Get();
      thicknessPriorPixel = thicknessPriorIterator.Get();
      
      outputPixel = phiPixel;
      thickness = vcl_sqrt((outputPixel - phiZeroPixel).GetSquaredNorm());

      for (i = 0; i < Dimension; i++)
        {
          phiPoint[i] = outputPixel[i];
        }

      for (i = 0; i < m_NumberOfSteps; i++)
        {
          if (velocityImage->TransformPhysicalPointToContinuousIndex(phiPoint, continousIndex) && thickness < thicknessPriorPixel)
            {
              vPixel = m_Interpolator->Evaluate(phiPoint);
              
              // Calculate Phi(x, t+dt) = Phi(x,t) + dt*v(phi(x,t), t)
              for (j = 0; j < Dimension; j++)
                {
                  outputPixel[j] = outputPixel[j] + factor * m_DeltaT*vPixel[j];
                  phiPoint[j] = phiPoint[j] + factor * m_DeltaT*vPixel[j];                  
                }
              thickness = vcl_sqrt((outputPixel - phiZeroPixel).GetSquaredNorm());
            }

        }          

      // Store the new value of Phi, and thickness
      
      outputIterator.Set(outputPixel);
      thicknessIterator.Set(thickness);
/*
      if ((phiIterator.GetIndex()[0] == 165 || phiIterator.GetIndex()[0] == 164) && phiIterator.GetIndex()[1] == 100)
        {
          niftkitkDebugMacro(<<"ThreadedGenerateData():index=" << phiIterator.GetIndex() \
            << ", started at:" << phiPixel \
            << ", finished at:" << outputPixel);  
        }
        */
    }
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


} // end namespace

#endif
