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
#ifndef __itkLocationToAngleImageFilter_txx
#define __itkLocationToAngleImageFilter_txx

#include "itkLocationToAngleImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkLogHelper.h"


namespace itk {

template <class TScalarType, unsigned int NDimensions>
LocationToAngleImageFilter<TScalarType, NDimensions>
::LocationToAngleImageFilter()
{
}

template <class TScalarType, unsigned int NDimensions>
void
LocationToAngleImageFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

template <class TScalarType, unsigned int NDimensions>
void
LocationToAngleImageFilter<TScalarType, NDimensions>
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
LocationToAngleImageFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const InputImageRegionType& outputRegionForThread, int threadNumber) 
{
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  // Get Pointers to images.
  typename InputImageType::Pointer inputImage 
    = static_cast< InputImageType * >( this->ProcessObject::GetInput(0) );

  typename OutputImageType::Pointer outputImage 
    = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  ImageRegionConstIteratorWithIndex< InputImageType >  inputIterator(inputImage, outputRegionForThread);
  ImageRegionIteratorWithIndex< OutputImageType > outputIterator(outputImage, outputRegionForThread);
  
  double squaredMagnitude = 0.;
  
  for (inputIterator.GoToBegin(), outputIterator.GoToBegin(); 
       !inputIterator.IsAtEnd(); 
       ++inputIterator, ++outputIterator)
    {
		outputIterator.Set( m_BodyAxisPoint1[0] );
    }

  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}


} // end namespace

#endif
