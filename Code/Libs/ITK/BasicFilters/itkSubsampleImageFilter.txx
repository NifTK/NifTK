/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSubsampleImageFilter_txx
#define __itkSubsampleImageFilter_txx

#include "itkSubsampleImageFilter.h"
#include <itkGaussianOperator.h>
#include <itkCastImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkExceptionObject.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>

#include <vnl/vnl_math.h>

#include <itkLogHelper.h>


namespace itk
{

/* ---------------------------------------------------------------------
   Constructor
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
SubsampleImageFilter<TInputImage, TOutputImage>
::SubsampleImageFilter()
{
  unsigned int i;

  m_MaximumError = 0.1;

  for (i = 0; i < ImageDimension; i++)
    {
      m_SubsamplingFactors[i] = 1.;
    }

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );
}
 
/* ---------------------------------------------------------------------
   GenerateData()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SubsampleImageFilter<TInputImage, TOutputImage>
::SetSubsamplingFactors(double data[]) 
{
  unsigned int i;

  for (i = 0; i < ImageDimension; i++)
    {
      m_SubsamplingFactors[i] = data[i];
    }
}


/* ---------------------------------------------------------------------
   GenerateData()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SubsampleImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  unsigned int idim;

  // Get the input and output pointers

  InputImageConstPointer  inputPtr = this->GetInput();

  // Create caster, smoother and resampleShrinker filters

  typedef CastImageFilter<TInputImage, TOutputImage>              CasterType;
  typedef DiscreteGaussianImageFilter<TOutputImage, TOutputImage> SmootherType;

  typedef ImageToImageFilter<TOutputImage,TOutputImage>           ImageToImageType;
  typedef ResampleImageFilter<TOutputImage,TOutputImage>          ResampleShrinkerType;

  typename CasterType::Pointer caster = CasterType::New();
  typename SmootherType::Pointer smoother = SmootherType::New();

  typename ResampleShrinkerType::Pointer shrinkerFilter;

  // Only one of these pointers is going to be valid, depending on the
  // value of UseShrinkImageFilter flag

  shrinkerFilter = ResampleShrinkerType::New();

  typedef itk::LinearInterpolateImageFunction< OutputImageType, double > LinearInterpolatorType;

  typename LinearInterpolatorType::Pointer interpolator =  LinearInterpolatorType::New();

  shrinkerFilter->SetInterpolator( interpolator );
  shrinkerFilter->SetDefaultPixelValue( 0 );

  
  // Setup the filters

  caster->SetInput( inputPtr );

  smoother->SetUseImageSpacing( false );
  smoother->SetInput( caster->GetOutput() );
  smoother->SetMaximumError( m_MaximumError );

  shrinkerFilter->SetInput( smoother->GetOutput() );

  double variance[ImageDimension];


  // Allocate memory for the output

  OutputImagePointer outputPtr = this->GetOutput( );

  outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
  outputPtr->Allocate();

  // compute shrink factors and variances
  
  for ( idim = 0; idim < ImageDimension; idim++ )
  {
    variance[idim] = vnl_math_sqr( 0.5 * static_cast<double>( m_SubsamplingFactors[idim] ) );
  }

  typedef itk::IdentityTransform<double,OutputImageType::ImageDimension> IdentityTransformType;

  typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();

  shrinkerFilter->SetOutputParametersFromImage( outputPtr );
  shrinkerFilter->SetTransform(identityTransform);


  // use mini-pipeline to compute output
  smoother->SetVariance( variance );

  shrinkerFilter->GraftOutput( outputPtr );

  // force to always update in case shrink factors are the same

  shrinkerFilter->Modified();
  shrinkerFilter->UpdateLargestPossibleRegion();

  this->GraftOutput( shrinkerFilter->GetOutput() );
}


/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SubsampleImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "MaximumError: " << m_MaximumError << std::endl;

  os << indent << "Subsampling factors: " << std::endl
     << m_SubsamplingFactors << std::endl;
}


/* ---------------------------------------------------------------------
   GenerateOutputInformation
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SubsampleImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{

  // call the superclass's implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  InputImageConstPointer inputPtr = this->GetInput();

  if ( !inputPtr  )
    itkExceptionMacro( << "Input has not been set" );

  const typename InputImageType::PointType& inputOrigin = inputPtr->GetOrigin();
  const typename InputImageType::SpacingType& inputSpacing = inputPtr->GetSpacing();
  const typename InputImageType::DirectionType& inputDirection = inputPtr->GetDirection();
  const typename InputImageType::SizeType& inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
  const typename InputImageType::IndexType& inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();

  typedef typename OutputImageType::SizeType  SizeType;
  typedef typename SizeType::SizeValueType    SizeValueType;
  typedef typename OutputImageType::IndexType IndexType;
  typedef typename IndexType::IndexValueType  IndexValueType;

  OutputImagePointer outputPtr;

  typename OutputImageType::PointType   outputOrigin;
  typename OutputImageType::SpacingType outputSpacing;

  SizeType    outputSize;
  IndexType   outputStartIndex;

  // We need to compute the output spacing, the output image size,
  // and the output image start index

  outputPtr = this->GetOutput( );

  if ( ! outputPtr  )
    itkExceptionMacro( << "Output has not been set" );

  for(unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ ) 
  {
    outputSpacing[idim] = inputSpacing[idim] * m_SubsamplingFactors[idim];

    outputSize[idim] = static_cast<SizeValueType>(
        vcl_floor(static_cast<double>(inputSize[idim]) / m_SubsamplingFactors[idim] ) );

    if( outputSize[idim] < 1 ) { outputSize[idim] = 1; }

    outputStartIndex[idim] = static_cast<IndexValueType>(
      vcl_ceil(static_cast<double>(inputStartIndex[idim]) / m_SubsamplingFactors[idim] ) );
  }
   
  // Now compute the new shifted origin for the updated levels;

  const typename OutputImageType::PointType::VectorType outputOriginOffset
    = ( inputDirection*(outputSpacing - inputSpacing) )*0.5;

  for ( unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ )
    outputOrigin[idim] = inputOrigin[idim] + outputOriginOffset[idim];

  typename OutputImageType::RegionType outputLargestPossibleRegion;

  outputLargestPossibleRegion.SetSize( outputSize );
  outputLargestPossibleRegion.SetIndex( outputStartIndex );

  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
  outputPtr->SetOrigin ( outputOrigin );
  outputPtr->SetSpacing( outputSpacing );
  outputPtr->SetDirection( inputDirection ); 
}


/* ---------------------------------------------------------------------
   GenerateInputRequestedRegion
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SubsampleImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  InputImagePointer  inputPtr = const_cast< InputImageType * >( this->GetInput() );

  if ( !inputPtr )
    {
    itkExceptionMacro( << "Input has not been set." );
    }

  // compute baseIndex and baseSize
  typedef typename OutputImageType::SizeType    SizeType;
  typedef typename SizeType::SizeValueType      SizeValueType;
  typedef typename OutputImageType::IndexType   IndexType;
  typedef typename IndexType::IndexValueType    IndexValueType;
  typedef typename OutputImageType::RegionType  RegionType;

  SizeType baseSize = this->GetOutput()->GetRequestedRegion().GetSize();
  IndexType baseIndex = this->GetOutput()->GetRequestedRegion().GetIndex();
  RegionType baseRegion;

  unsigned int idim;
  for( idim = 0; idim < ImageDimension; idim++ )
    {
    baseIndex[idim] *= static_cast<IndexValueType>( m_SubsamplingFactors[idim] );
    baseSize[idim] *= static_cast<SizeValueType>( m_SubsamplingFactors[idim] );
    }

  baseRegion.SetIndex( baseIndex );
  baseRegion.SetSize( baseSize );

  // compute requirements for the smoothing part
  typedef typename TOutputImage::PixelType                 OutputPixelType;
  typedef GaussianOperator<OutputPixelType,ImageDimension> OperatorType;

  OperatorType *oper = new OperatorType;

  typename TInputImage::SizeType radius;

  RegionType inputRequestedRegion = baseRegion;

  for( idim = 0; idim < TInputImage::ImageDimension; idim++ )
    {
    oper->SetDirection(idim);
    oper->SetVariance( vnl_math_sqr( 0.5 * static_cast<float>( m_SubsamplingFactors[idim] ) ) );
    oper->SetMaximumError( m_MaximumError );
    oper->CreateDirectional();
    radius[idim] = oper->GetRadius()[idim];
    }
  delete oper;

  inputRequestedRegion.PadByRadius( radius );

  // make sure the requested region is within the largest possible
  inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() );

  // set the input requested region
  inputPtr->SetRequestedRegion( inputRequestedRegion );
}


} // namespace itk

#endif
