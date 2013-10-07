/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkInterpolateVectorFieldFilter_txx
#define __itkInterpolateVectorFieldFilter_txx

#include "itkInterpolateVectorFieldFilter.h"
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkProgressReporter.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <itkLogHelper.h>

namespace itk
{
/**
 * Initialize new instance
 */
template <class TScalarType, unsigned int NDimensions>
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::InterpolateVectorFieldFilter()
{
  m_Interpolator = VectorLinearInterpolateImageFunction<InputImageType, TScalarType>::New();
  m_DefaultPixelValue.Fill(0); 
  niftkitkDebugMacro(<<"InterpolateVectorFieldFilter():Constructed");
}

/**
 * Print out a description of self
 *
 */
template <class TScalarType, unsigned int NDimensions> 
void 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Interpolator: " << m_Interpolator << std::endl;
  os << indent << "DefaultPixelValue: " << m_DefaultPixelValue << std::endl;
  return;
}

template <class TScalarType, unsigned int NDimensions>
void
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::VerifyInputInformation()
{
  // Do nothing for now.
}

template <class TScalarType, unsigned int NDimensions>
void
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::SetNthInput(unsigned int idx, const InputImageType *image)
{
  this->ProcessObject::SetNthInput(idx, const_cast< InputImageType* >(image));
  this->Modified();
  
  niftkitkDebugMacro(<<"SetNthInput(" << idx << ", " << image << ")" );
}

/**
 * Set up state of filter before multi-threading.
 * InterpolatorType::SetInputImage is not thread-safe and hence
 * has to be set up before ThreadedGenerateData
 */
template <class TScalarType, unsigned int NDimensions> 
void 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{

  if( !m_Interpolator )
    {
      niftkitkExceptionMacro(<< "Interpolator not set");
    }

  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  
  // We should have exactly 2 inputs.
  if (numberOfInputs != 2)
    {
      niftkitkExceptionMacro(<< "InterpolateVectorFieldFilter should have exactly 2 inputs.");
    }

  // Connect input image to interpolator
  m_Interpolator->SetInputImage( this->GetInput(0) );

  niftkitkDebugMacro(<<"BeforeThreadedGenerateData():Interpolator set to image:" << this->GetInput(0));
}

/**
 * Set up state of filter after multi-threading.
 */
template <class TScalarType, unsigned int NDimensions> 
void 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::AfterThreadedGenerateData()
{
  // Disconnect input image from the interpolator
  m_Interpolator->SetInputImage( NULL );
  
  niftkitkDebugMacro(<<"AfterThreadedGenerateData():Interpolator disconnected");

}

/**
 * ThreadedGenerateData
 */
template <class TScalarType, unsigned int NDimensions> 
void 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::ThreadedGenerateData(
  const OutputImageRegionType& outputRegionForThread,
  ThreadIdType threadId)
{
  niftkitkDebugMacro(<<"ThreadedGenerateData():Executing thread:" << threadId);

  // Get the image pointers. The output image is guaranteed to be the same
  // dimension as the input image 1. So we iterate over the output region.
  
  OutputImagePointer      outputPtr = this->GetOutput();
  InputImageConstPointer  interpolatedPtr = this->GetInput(0);
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():interpolatedPtr=" << interpolatedPtr.GetPointer() << ", outputPtr=" << outputPtr.GetPointer());
  
  // Create an iterator that will walk the output region for this thread.
  typedef ImageRegionIteratorWithIndex<OutputImageType> OutputIterator;
  OutputIterator outIt(outputPtr, outputRegionForThread);

  // Define a few indices that will be used to translate from an input pixel
  // to an output pixel
  PointType outputPoint;         // Coordinates of current output pixel

  typedef ContinuousIndex<TScalarType, NDimensions> ContinuousIndexType;
  ContinuousIndexType inputIndex;

  // Support for progress methods/callbacks
  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
        
  typedef typename InterpolatorType::OutputType OutputType;

  // Walk the output region
  outIt.GoToBegin();

  OutputPixelType pixval; 
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Starting loop");
  
  while ( !outIt.IsAtEnd() )
    {
      // Determine the index of the current output pixel
      outputPtr->TransformIndexToPhysicalPoint( outIt.GetIndex(), outputPoint );

      // Compute corresponding input pixel position
      interpolatedPtr->TransformPhysicalPointToContinuousIndex(outputPoint, inputIndex);

      // Just for if you are debugging.
      // niftkitkDebugMacro(<<"outputPoint:" << outputPoint << ", inputIndex:" << inputIndex);
        
      // Evaluate input at right position and copy to the output
      if( m_Interpolator->IsInsideBuffer(inputIndex) )
        {
          
          pixval = m_Interpolator->EvaluateAtContinuousIndex( inputIndex );
          outIt.Set( pixval );
        }
      else
        {
          outIt.Set(m_DefaultPixelValue); // default background value
        }

      progress.CompletedPixel();
      ++outIt;
    }
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished loop");
  
  return;
}

/** 
 * Inform pipeline of necessary input image region
 */
template <class TScalarType, unsigned int NDimensions> 
void 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::GenerateInputRequestedRegion()
{
  // call the superclass's implementation of this method
  Superclass::GenerateInputRequestedRegion();

  if ( !this->GetInput(0) || !this->GetInput(1) )
    {
      niftkitkDebugMacro(<<"GenerateInputRequestedRegion():Returning because inputs are missing input(0)=" << this->GetInput(0) << ", input(1)=" << this->GetInput(1));
      return;
    }

  // get pointers to the input and output
  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++)
    {
    
      InputImagePointer input = const_cast< InputImageType *>( this->GetInput(i) );

      // Request the entire input image 
      InputImageRegionType inputRegion;
      inputRegion = input->GetLargestPossibleRegion();
      input->SetRequestedRegion(inputRegion);
      
      niftkitkDebugMacro(<<"GenerateInputRequestedRegion():Requested largest region on image(" << i << ")=" << input.GetPointer());
    }
    
  return;
}

/** 
 * Inform pipeline of required output region
 */
template <class TScalarType, unsigned int NDimensions> 
void 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::GenerateOutputInformation()
{
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  OutputImagePointer outputPtr = this->GetOutput();
  if ( !outputPtr )
    {
      niftkitkDebugMacro(<<"GenerateOutputInformation():Returning as output is null???");
      return;
    }

  // Set the size of the output region. Should be exactly as input(1).
  InputImagePointer input = const_cast< InputImageType *>( this->GetInput(1) );
  
  OutputImageRegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize( input->GetLargestPossibleRegion().GetSize() );
  outputLargestPossibleRegion.SetIndex( input->GetLargestPossibleRegion().GetIndex() );
  
  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
  niftkitkDebugMacro(<<"GenerateOutputInformation():LargestPossibleRegion=" << outputPtr->GetLargestPossibleRegion().GetSize());
  
  outputPtr->SetSpacing( input->GetSpacing() );
  niftkitkDebugMacro(<<"GenerateOutputInformation():Spacing=" << outputPtr->GetSpacing());
  
  outputPtr->SetOrigin( input->GetOrigin() );
  niftkitkDebugMacro(<<"GenerateOutputInformation():Origin=" << outputPtr->GetOrigin());
  
  outputPtr->SetDirection( input->GetDirection() );
  return;
}

/** 
 * Verify if any of the components has been modified.
 */
template <class TScalarType, unsigned int NDimensions> 
unsigned long 
InterpolateVectorFieldFilter<TScalarType, NDimensions>
::GetMTime( void ) const
{
  unsigned long latestTime = Object::GetMTime(); 

  if( m_Interpolator )
    {
    if( latestTime < m_Interpolator->GetMTime() )
      {
      latestTime = m_Interpolator->GetMTime();
      }
    }

  niftkitkDebugMacro(<<"GetMTime():Returns=" << latestTime);
  return latestTime;
}

} // end namespace itk

#endif
