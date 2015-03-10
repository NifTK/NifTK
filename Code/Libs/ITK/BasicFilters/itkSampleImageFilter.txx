/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSampleImageFilter_txx
#define __itkSampleImageFilter_txx

#include "itkSampleImageFilter.h"
#include "itkLewisGriffinRecursiveGaussianImageFilter.h"

#include <itkCastImageFilter.h>
#include <itkExceptionObject.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>
#include <itkWindowedSincInterpolateImageFunction.h>

#include <vnl/vnl_math.h>

#include <itkLogHelper.h>


namespace itk
{

/* ---------------------------------------------------------------------
   Constructor
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
SampleImageFilter<TInputImage, TOutputImage>
::SampleImageFilter()
{
  unsigned int i;

  m_Verbose = false;

  m_IsotropicVoxels = false;

  for (i = 0; i < ImageDimension; i++)
  {
    m_SamplingFactors[i] = 1.;
  }

  m_Interpolation = LINEAR;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );
}
 
/* ---------------------------------------------------------------------
   SetSamplingFactors()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SampleImageFilter<TInputImage, TOutputImage>
::SetSamplingFactors(double data[]) 
{
  unsigned int i;

  for (i = 0; i < ImageDimension; i++)
  {
    m_SamplingFactors[i] = data[i];
  }
}


/* ---------------------------------------------------------------------
   SetSamplingFactors()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SampleImageFilter<TInputImage, TOutputImage>
::SetSamplingFactors( itk::Array< double > &sampling ) 
{
  unsigned int i;

  for (i = 0; i < ImageDimension; i++)
  {
    m_SamplingFactors[i] = sampling[i];
  }
}


/* ---------------------------------------------------------------------
   GetSmoothedImage()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
typename SampleImageFilter<TInputImage, TOutputImage>::OutputImagePointer
SampleImageFilter<TInputImage, TOutputImage>
::GetSmoothedImage( unsigned int idim, OutputImagePointer image )
{
  
  // Sampling

  if ( m_SamplingFactors[idim] > 1 )
  {
    typename OutputImageType::SpacingType spacing = image->GetSpacing();

    double mmStdDev;  // The standard deviation of the Gaussian to use
    double mmScale;   // The scale in mm of the smoothing

    mmStdDev = m_SamplingFactors[idim]*spacing[idim]/(2.*sqrt(3.));
    mmScale = mmStdDev*mmStdDev/2.;

    if ( this->GetVerbose() ) 
    {
      std::cout << "Dimension: " << idim 
                << "  Spacing: " << spacing[idim]
                << "mm, sampling: " << m_SamplingFactors[idim]
                << ", Gaussian standard deviation: " << mmStdDev << "mm"
                << " scale: " << mmScale << "mm" << std::endl; 
    }

    typedef LewisGriffinRecursiveGaussianImageFilter<TOutputImage, TOutputImage> SmootherType;

    typename SmootherType::Pointer smoother = SmootherType::New();

    smoother->SetSingleThreadedExecution();
    smoother->SetSigma( mmScale );
    smoother->SetInput( image );
    smoother->SetZeroOrder();
    smoother->SetDirection( idim );

    smoother->Update();
    
    return smoother->GetOutput();
  }

  // Supersampling so no smoothing

  else
  {
    return image;
  }

  return 0;
}


/* ---------------------------------------------------------------------
   GenerateData()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SampleImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  unsigned int idim;

  // Get the input and output pointers

  InputImageConstPointer  imInput = this->GetInput();


  // Allocate memory for the output

  OutputImagePointer imOutput = this->GetOutput( );

  imOutput->SetBufferedRegion( imOutput->GetRequestedRegion() );
  imOutput->Allocate();


  // Create caster

  typedef CastImageFilter<TInputImage, TOutputImage> CasterType;

  typename CasterType::Pointer caster = CasterType::New();

  caster->SetInput( imInput );
  caster->Update();


  // Create the smoothing pipeline


  OutputImagePointer imSmoothingPipeline = caster->GetOutput();;

  for ( idim = 0; idim < ImageDimension; idim++ )
  {
    imSmoothingPipeline = GetSmoothedImage( idim, imSmoothingPipeline );
  }

  
  // The resampling filter

  typedef ResampleImageFilter<TOutputImage,TOutputImage> ResampleType;

  typename ResampleType::Pointer resampleFilter = ResampleType::New();

  switch ( m_Interpolation )
  {
  case NEAREST: 
  {
    typedef itk::NearestNeighborInterpolateImageFunction< TOutputImage, double> NearestNeighbourInterpolatorType;

    typename NearestNeighbourInterpolatorType::Pointer interpolator 
      = NearestNeighbourInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  case LINEAR: 
  {
    typedef itk::LinearInterpolateImageFunction< TOutputImage, double >         LinearInterpolatorType;

    typename LinearInterpolatorType::Pointer interpolator 
      = LinearInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  case BSPLINE: 
  {
    typedef itk::BSplineInterpolateImageFunction< TOutputImage, double >        BSplineInterpolatorType;

    typename BSplineInterpolatorType::Pointer interpolator 
      = BSplineInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  case SINC: 
  {
    typedef itk::ConstantBoundaryCondition< TOutputImage > BoundaryConditionType;
    const static unsigned int WindowRadius = 5;
    typedef itk::Function::WelchWindowFunction<WindowRadius>  WindowFunctionType;
    typedef itk::WindowedSincInterpolateImageFunction< TOutputImage, 
                                                       WindowRadius,  
                                                       WindowFunctionType,  
                                                       BoundaryConditionType,  
                                                       double  > SincInterpolatorType;

    typename SincInterpolatorType::Pointer interpolator 
      = SincInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  default:
  {
    itkExceptionMacro( << "Interpolator type not recognised." );
  }
  }

  resampleFilter->SetDefaultPixelValue( 0 );

  resampleFilter->SetInput( imSmoothingPipeline );

  typedef itk::IdentityTransform<double,OutputImageType::ImageDimension> IdentityTransformType;

  typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();

  resampleFilter->SetOutputParametersFromImage( imOutput );
  resampleFilter->SetTransform(identityTransform);

  resampleFilter->GraftOutput( imOutput );

  // force to always update in case shrink factors are the same

  resampleFilter->Modified();
  resampleFilter->UpdateLargestPossibleRegion();

  this->GraftOutput( resampleFilter->GetOutput() );
}


/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SampleImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Sampling factors: " << std::endl
     << m_SamplingFactors << std::endl;
}


/* ---------------------------------------------------------------------
   GenerateOutputInformation
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SampleImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{

  // call the superclass's implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  InputImageConstPointer imInput = this->GetInput();

  if ( !imInput  )
    itkExceptionMacro( << "Input has not been set" );

  const typename InputImageType::PointType& inputOrigin = imInput->GetOrigin();
  const typename InputImageType::SpacingType& inputSpacing = imInput->GetSpacing();
  const typename InputImageType::DirectionType& inputDirection = imInput->GetDirection();
  const typename InputImageType::SizeType& inputSize = imInput->GetLargestPossibleRegion().GetSize();
  const typename InputImageType::IndexType& inputStartIndex = imInput->GetLargestPossibleRegion().GetIndex();

  typedef typename OutputImageType::SizeType  SizeType;
  typedef typename SizeType::SizeValueType    SizeValueType;
  typedef typename OutputImageType::IndexType IndexType;
  typedef typename IndexType::IndexValueType  IndexValueType;

  // Isotropic voxels?

  if ( m_IsotropicVoxels )
  {
    // Calculate the minimum spacing
    double minSpacing = std::numeric_limits<double>::max();

    for (unsigned int j = 0; j < InputImageType::ImageDimension; j++)
    {
      if ( inputSpacing[j] < minSpacing )
      {
        minSpacing = inputSpacing[j];
      }
    }

    // Calculate the subsampling factors    
    for (unsigned int j = 0; j < InputImageType::ImageDimension; j++)
    {
      m_SamplingFactors[j] = minSpacing/inputSpacing[j];
    }
  }

  OutputImagePointer imOutput;

  typename OutputImageType::PointType   outputOrigin;
  typename OutputImageType::SpacingType outputSpacing;

  SizeType    outputSize;
  IndexType   outputStartIndex;

  // We need to compute the output spacing, the output image size,
  // and the output image start index

  imOutput = this->GetOutput( );

  if ( ! imOutput  )
  {
    itkExceptionMacro( << "Output has not been set" );
  }

  for (unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ ) 
  {
    outputSize[idim] = static_cast<SizeValueType>(
      vcl_floor(static_cast<double>(inputSize[idim]) / m_SamplingFactors[idim] + 0.5) );

    if ( outputSize[idim] < 1 ) { 
      outputSize[idim] = 1; 
    }

    // Ensure the dimensions of the images are identical

    outputSpacing[idim] = static_cast<double>(inputSize[idim])*inputSpacing[idim]
      / static_cast<double>(outputSize[idim]);



    outputStartIndex[idim] = static_cast<IndexValueType>(
      vcl_floor(static_cast<double>(inputStartIndex[idim])*inputSpacing[idim] 
                / outputSpacing[idim] + 0.5 ));
   
    if ( this->GetVerbose() ) 
    {
      std::cout << "Sampling: " << idim 
                << std::setw( 6 ) << inputSize[idim] << "voxels, "
                << std::setw( 8 ) << inputSpacing[idim] << "mm by " 
                << std::setw( 6 ) << m_SamplingFactors[idim] << " -> "
                << std::setw( 8 ) << outputSpacing[idim] << "mm, "         
                << std::setw( 6 ) << outputSize[idim] << "voxels"
                << std::endl; 
    }
  }

  if ( this->GetVerbose() ) 
  {
    std::cout << " Start: ";

    for (unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ ) 
    {
      std::cout << std::setw( 6 ) << inputStartIndex[idim];

      if ( idim < OutputImageType::ImageDimension - 1 )
      {
        std::cout << ", ";
      }
    }

    std::cout << " -> ";
    
    for (unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ ) 
    {
      std::cout << std::setw( 6 ) << outputStartIndex[idim];

      if ( idim < OutputImageType::ImageDimension - 1 )
      {
        std::cout << ", ";
      }
    }
    std::cout << std::endl; 
  }


  // Now compute the new shifted origin for the updated levels;

  const typename OutputImageType::PointType::VectorType outputOriginOffset
    = ( inputDirection*(outputSpacing - inputSpacing) )*0.5;

  for ( unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ )
  {
    outputOrigin[idim] = inputOrigin[idim] + outputOriginOffset[idim];
  }

  if ( this->GetVerbose() ) 
  {
    std::cout << " Origin: ";

    for (unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ ) 
    {
      std::cout << std::setw( 6 ) << inputOrigin[idim];

      if ( idim < OutputImageType::ImageDimension - 1 )
      {
        std::cout << ", ";
      }
    }

    std::cout << " -> ";
    
    for (unsigned int idim = 0; idim < OutputImageType::ImageDimension; idim++ ) 
    {
      std::cout << std::setw( 6 ) << outputOrigin[idim];

      if ( idim < OutputImageType::ImageDimension - 1 )
      {
        std::cout << ", ";
      }
    }
    std::cout << std::endl; 
  }

  typename OutputImageType::RegionType outputLargestPossibleRegion;

  outputLargestPossibleRegion.SetSize( outputSize );
  outputLargestPossibleRegion.SetIndex( outputStartIndex );

  imOutput->SetLargestPossibleRegion( outputLargestPossibleRegion );
  imOutput->SetOrigin ( outputOrigin );
  imOutput->SetSpacing( outputSpacing );
  imOutput->SetDirection( inputDirection ); 
}


/* ---------------------------------------------------------------------
   GenerateInputRequestedRegion
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
SampleImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  InputImagePointer  imInput = const_cast< InputImageType * >( this->GetInput() );

  if ( !imInput )
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
    baseIndex[idim] *= static_cast<IndexValueType>( m_SamplingFactors[idim] );
    baseSize[idim] *= static_cast<SizeValueType>( m_SamplingFactors[idim] );
  }

  baseRegion.SetIndex( baseIndex );
  baseRegion.SetSize( baseSize );


  // make sure the requested region is within the largest possible
  baseRegion.Crop( imInput->GetLargestPossibleRegion() );

  // set the input requested region
  imInput->SetRequestedRegion( baseRegion );
}


} // namespace itk

#endif
