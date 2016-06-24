/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkNormaliseVectorFilter_txx
#define __itkNormaliseVectorFilter_txx

#include "itkNormaliseVectorFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
NormaliseVectorFilter<TScalarType, NDimensions>
::NormaliseVectorFilter()
{
  m_Normalise = true;
  m_LengthTolerance = 0.0000001;
  niftkitkDebugMacro(<<"NormaliseVectorFilter():m_Normalise=" << m_Normalise \
      << ", m_LengthTolerance=" << m_LengthTolerance \
      );
}

template <class TScalarType, unsigned int NDimensions>
void
NormaliseVectorFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Normalise = " << m_Normalise << std::endl;
  os << indent << "Tolerance = " << m_LengthTolerance << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void
NormaliseVectorFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, ThreadIdType threadNumber)
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber \
      << ", m_Normalise=" << m_Normalise \
      << ", m_LengthTolerance=" << m_LengthTolerance \
      );

  // Get Pointers to images.
  typename InputImageType::Pointer inputImage 
    = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionConstIteratorWithIndex<InputImageType> inputIterator(inputImage, outputRegionForThread);
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputRegionForThread);
  
  double length;
  OutputPixelType vector;
  OutputPixelType zero;
  zero.Fill(0);
  
  for (inputIterator.GoToBegin(),
       outputIterator.GoToBegin();
       !inputIterator.IsAtEnd();
       ++inputIterator,
       ++outputIterator)
    {
      if (m_Normalise)
        {
          vector = inputIterator.Get();
          length = vector.GetNorm();
          if (length > m_LengthTolerance)
            {
              vector /= length;
            }
          else
            {
              vector = zero;
            }
          outputIterator.Set(vector);
        }
      else
        {
          outputIterator.Set(inputIterator.Get());
        }
    }
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


} // end namespace

#endif
