/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkAddUpdateToTimeVaryingVelocityFieldFilter_txx
#define __itkAddUpdateToTimeVaryingVelocityFieldFilter_txx

#include "itkAddUpdateToTimeVaryingVelocityFieldFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"
#include "itkUCLMacro.h"
#include "itkLogHelper.h"

namespace itk {

template <class TScalarType, unsigned int NDimensions>
AddUpdateToTimeVaryingVelocityFieldFilter<TScalarType, NDimensions>
::AddUpdateToTimeVaryingVelocityFieldFilter()
{
  m_TimePoint = 0;
  m_ScaleFactor = 1;
  m_MaxDeformation = 0;
  m_OverWrite = false;
  
  m_UpdateImage = UpdateImageType::New();
  m_UpdateImage = NULL;
  
  m_UpdateInverseImage = UpdateImageType::New();
  m_UpdateInverseImage = NULL;
  
  this->SetInPlace(false);
  
}

template <class TScalarType, unsigned int NDimensions>
void
AddUpdateToTimeVaryingVelocityFieldFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "TimePoint=" << m_TimePoint << std::endl;
  os << indent << "InPlace=" << this->GetInPlace() << std::endl;
  os << indent << "ScaleFactor=" << m_ScaleFactor << std::endl;
  os << indent << "MaxDeformation=" << m_MaxDeformation << std::endl;
  os << indent << "OverWrite=" << m_OverWrite << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void 
AddUpdateToTimeVaryingVelocityFieldFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{
  // Check that update images have same size (apart from time dimension).
  TimeVaryingVelocitySizeType velocityFieldSize = this->GetOutput()->GetLargestPossibleRegion().GetSize();

  if (m_UpdateImage.IsNull())
  {
    niftkitkExceptionMacro(<< "Update Image is null, so this filter can't run.");
  }
  
  UpdateImageSizeType updateFieldSize = m_UpdateImage->GetLargestPossibleRegion().GetSize();
  
  for (unsigned int i = 0; i < NDimensions; i++)
  {
    if (velocityFieldSize[i] != updateFieldSize[i])
    {
      niftkitkExceptionMacro(<< "For update image: velocityFieldSize[" << i << "]=" << velocityFieldSize[i] \
        << " != updateFieldSize[" << i << "]=" << updateFieldSize[i]);
    }
  }
  
  if (m_UpdateInverseImage)
  {
    updateFieldSize = m_UpdateInverseImage->GetLargestPossibleRegion().GetSize();
    for (unsigned int i = 0; i < NDimensions; i++)
      {
        if (velocityFieldSize[i] != updateFieldSize[i])
          {
            niftkitkExceptionMacro(<< "For update inverse image: velocityFieldSize[" << i << "]=" << velocityFieldSize[i] \
              << " != updateFieldSize[" << i << "]=" << updateFieldSize[i]);
          }
      }
  }
}

template <class TScalarType, unsigned int NDimensions>
void
AddUpdateToTimeVaryingVelocityFieldFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const TimeVaryingVelocityRegionType& regionForThread, int threadNumber) 
{
  
  typename TimeVaryingVelocityImageType::Pointer inputImage = static_cast< TimeVaryingVelocityImageType * >(this->ProcessObject::GetInput(0));
  typename TimeVaryingVelocityImageType::Pointer outputImage = static_cast< TimeVaryingVelocityImageType * >(this->ProcessObject::GetOutput(0));
  
  TimeVaryingVelocitySizeType velocityFieldSize = inputImage->GetLargestPossibleRegion().GetSize();
  int timePointToUpdate = (unsigned int) (((float) velocityFieldSize[NDimensions] - 1.0) * m_TimePoint + 0.5);
  
  niftkitkDebugMacro(<< "ThreadedGenerateData():Started thread:" << threadNumber \
    << ", velocityFieldSize=" << velocityFieldSize \
    << ", m_TimePoint=" << m_TimePoint \
    << ", timePointToUpdate=" << timePointToUpdate \
    );

  TimeVaryingVelocityIndexType velocityIndex;
  UpdateImageIndexType updateIndex;
    
  ImageRegionIteratorWithIndex<TimeVaryingVelocityImageType> inputIterator(inputImage, regionForThread);
  ImageRegionIteratorWithIndex<TimeVaryingVelocityImageType> outputIterator(outputImage, regionForThread);
  
  for (inputIterator.GoToBegin(),
       outputIterator.GoToBegin();
       !inputIterator.IsAtEnd();
       ++inputIterator,
       ++outputIterator)
    {
      velocityIndex = inputIterator.GetIndex();
      
      if (velocityIndex[NDimensions] == timePointToUpdate)
        {        
          for (unsigned int i = 0; i < NDimensions; i++)
            {
              updateIndex[i] = velocityIndex[i];
            }

          if (m_UpdateImage.IsNotNull())
            {
              if (m_OverWrite)
              {
                outputIterator.Set(m_ScaleFactor * (m_UpdateImage->GetPixel(updateIndex)));
              }
              else
              {
                outputIterator.Set(inputIterator.Get() + m_ScaleFactor * (m_UpdateImage->GetPixel(updateIndex)));
              }
            }
               
          if (m_UpdateInverseImage.IsNotNull())
            {
              if (m_OverWrite)
              {
                outputIterator.Set(m_ScaleFactor * (m_UpdateImage->GetPixel(updateIndex)));
              }
              else
              {
                outputIterator.Set(outputIterator.Get() - m_ScaleFactor * (m_UpdateInverseImage->GetPixel(updateIndex)));
              }
            }
        }
        else
        {
          outputIterator.Set(inputIterator.Get());  
        }
    }
    
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}

template <class TScalarType, unsigned int NDimensions>
void 
AddUpdateToTimeVaryingVelocityFieldFilter<TScalarType, NDimensions>
::AfterThreadedGenerateData()
{
  m_MaxDeformation = 0;
  double norm;
  
  typename TimeVaryingVelocityImageType::Pointer outputImage = static_cast< TimeVaryingVelocityImageType * >(this->ProcessObject::GetOutput(0));
  ImageRegionConstIterator<TimeVaryingVelocityImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
  {
    norm = outputIterator.Get().GetNorm();
    if (norm > m_MaxDeformation)
    {
      m_MaxDeformation = norm;
    }
  }
}

} // end namespace
#endif
