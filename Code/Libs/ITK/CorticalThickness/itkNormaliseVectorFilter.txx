/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkNormaliseVectorFilter_txx
#define __itkNormaliseVectorFilter_txx

#include "itkNormaliseVectorFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

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
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
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
