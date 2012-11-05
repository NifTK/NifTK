/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkRegionGrowSurfacePoints.txx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkRegionGrowSurfacePoints_txx
#define __itkRegionGrowSurfacePoints_txx

#include "itkRegionGrowSurfacePoints.h"
#include "itkBinaryThresholdSurfaceVoxelImageFunction.h"
#include "itkFloodFilledImageFunctionConditionalIterator.h"
#include "itkProgressReporter.h"

#ifdef ITK_USE_REVIEW
#include "itkShapedFloodFilledImageFunctionConditionalIterator.h"
#endif

namespace itk
{

/**
 * Constructor
 */
template <class TInputImage, class TOutputImage>
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::RegionGrowSurfacePoints()
{
  m_Lower = NumericTraits<InputImagePixelType>::NonpositiveMin();
  m_Upper = NumericTraits<InputImagePixelType>::max();
  m_ReplaceValue = NumericTraits<OutputImagePixelType>::One;

  typename InputPixelObjectType::Pointer lower = InputPixelObjectType::New();
  lower->Set( NumericTraits< InputImagePixelType >::NonpositiveMin() );
  this->ProcessObject::SetNthInput( 1, lower );

  typename InputPixelObjectType::Pointer upper = InputPixelObjectType::New();
  upper->Set( NumericTraits< InputImagePixelType >::max() );
  this->ProcessObject::SetNthInput( 2, upper );
}

template <class TInputImage, class TOutputImage>
void
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::SetSeed ( const IndexType & seed )
{
  this->ClearSeeds();
  this->AddSeed ( seed );
}

template <class TInputImage, class TOutputImage>
void
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::AddSeed(const IndexType & seed)
{
  this->m_SeedList.push_back ( seed );
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::ClearSeeds ()
{
  if( m_SeedList.size() > 0 )
    {
    this->m_SeedList.clear();
    this->Modified();
    }
}

/**
 * Standard PrintSelf method.
 */
template <class TInputImage, class TOutputImage>
void
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Upper: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_Upper)
     << std::endl;
  os << indent << "Lower: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_Lower)
     << std::endl;
  os << indent << "ReplaceValue: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_ReplaceValue)
     << std::endl;
}

template <class TInputImage, class TOutputImage>
void 
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
  if ( this->GetInput() )
    {
    InputImagePointer image = 
      const_cast< InputImageType * >( this->GetInput() );
    image->SetRequestedRegionToLargestPossibleRegion();
    }
}

template <class TInputImage, class TOutputImage>
void 
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  Superclass::EnlargeOutputRequestedRegion(output);
  output->SetRequestedRegionToLargestPossibleRegion();
}

template <class TInputImage, class TOutputImage>
void 
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::SetLowerInput( const InputPixelObjectType * input )
{
  if (input != this->GetLowerInput())
    {
    this->ProcessObject::SetNthInput(1,
                                     const_cast<InputPixelObjectType*>(input));
    this->Modified();
    }
}


template <class TInputImage, class TOutputImage>
void 
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::SetUpperInput( const InputPixelObjectType * input )
{
  if (input != this->GetUpperInput())
    {
    this->ProcessObject::SetNthInput(2,
                                     const_cast<InputPixelObjectType*>(input));
    this->Modified();
    }
}

template <class TInputImage, class TOutputImage>
void
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::SetUpper(const InputImagePixelType threshold)
{
  // first check to see if anything changed
  typename InputPixelObjectType::Pointer upper=this->GetUpperInput();
  if (upper && upper->Get() == threshold)
    {
    return;
    }
  
  // create a data object to use as the input and to store this
  // threshold. we always create a new data object to use as the input
  // since we do not want to change the value in any current input
  // (the current input could be the output of another filter or the
  // current input could be used as an input to several filters)
  upper = InputPixelObjectType::New();
  this->ProcessObject::SetNthInput(2, upper);

  upper->Set(threshold);
  this->Modified();
}


template <class TInputImage, class TOutputImage>
void
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::SetLower(const InputImagePixelType threshold)
{
  // first check to see if anything changed
  typename InputPixelObjectType::Pointer lower=this->GetLowerInput();
  if (lower && lower->Get() == threshold)
    {
    return;
    }
  
  // create a data object to use as the input and to store this
  // threshold. we always create a new data object to use as the input
  // since we do not want to change the value in any current input
  // (the current input could be the output of another filter or the
  // current input could be used as an input to several filters)
  lower = InputPixelObjectType::New();
  this->ProcessObject::SetNthInput(1, lower);

  lower->Set(threshold);
  this->Modified();
}


template <class TInputImage, class TOutputImage>
typename RegionGrowSurfacePoints<TInputImage, TOutputImage>::InputPixelObjectType *
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::GetLowerInput()
{
  typename InputPixelObjectType::Pointer lower
    = static_cast<InputPixelObjectType *>(this->ProcessObject::GetInput(1));
  if (!lower)
    {
    // no input object available, create a new one and set it to the
    // default threshold
    lower = InputPixelObjectType::New();
    lower->Set( NumericTraits<InputImagePixelType>::NonpositiveMin() );
    this->ProcessObject::SetNthInput( 1, lower );
    }
    
  return lower;
}


template <class TInputImage, class TOutputImage>
typename RegionGrowSurfacePoints<TInputImage, TOutputImage>::InputPixelObjectType *
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::GetUpperInput()
{
  typename InputPixelObjectType::Pointer upper
    = static_cast<InputPixelObjectType *>(this->ProcessObject::GetInput(2));
  if (!upper)
    {
    // no input object available, create a new one and set it to the
    // default threshold
    upper = InputPixelObjectType::New();
    upper->Set( NumericTraits<InputImagePixelType>::NonpositiveMin() );
    this->ProcessObject::SetNthInput( 2, upper );
    }
    
  return upper;
}


template <class TInputImage, class TOutputImage>
typename RegionGrowSurfacePoints<TInputImage, TOutputImage>::InputImagePixelType
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::GetLower() const
{
  typename InputPixelObjectType::Pointer lower
    = const_cast<Self*>(this)->GetLowerInput();

  return lower->Get();
}

template <class TInputImage, class TOutputImage>
typename RegionGrowSurfacePoints<TInputImage, TOutputImage>::InputImagePixelType
RegionGrowSurfacePoints<TInputImage, TOutputImage>
::GetUpper() const
{
  typename InputPixelObjectType::Pointer upper
    = const_cast<Self*>(this)->GetUpperInput();

  return upper->Get();
}

template <class TInputImage, class TOutputImage>
void 
RegionGrowSurfacePoints<TInputImage,TOutputImage>
::GenerateData()
{
  InputImageConstPointer inputImage = this->GetInput();
  OutputImagePointer outputImage = this->GetOutput();

  typename InputPixelObjectType::Pointer lowerThreshold=this->GetLowerInput();
  typename InputPixelObjectType::Pointer upperThreshold=this->GetUpperInput();

  m_Lower = lowerThreshold->Get();
  m_Upper = upperThreshold->Get();
  
  // Zero the output
  OutputImageRegionType region =  outputImage->GetRequestedRegion();
  outputImage->SetBufferedRegion( region );
  outputImage->Allocate();
  outputImage->FillBuffer ( NumericTraits<OutputImagePixelType>::Zero );
  
  typedef BinaryThresholdSurfaceVoxelImageFunction<InputImageType, double> FunctionType;
  
  typename FunctionType::Pointer function = FunctionType::New();
  function->SetInputImage ( inputImage );
  function->ThresholdBetween ( m_Lower, m_Upper );
  
  ProgressReporter progress(this, 0, region.GetNumberOfPixels());
  
  typedef ShapedFloodFilledImageFunctionConditionalIterator<OutputImageType, FunctionType> IteratorType;
  IteratorType it ( outputImage, function, m_SeedList );
  it.FullyConnectedOn();
  it.GoToBegin();
  
  while( !it.IsAtEnd())
  {
    it.Set(m_ReplaceValue);
    ++it;
    progress.CompletedPixel();  // potential exception thrown here
  }
}


} // end namespace itk

#endif
