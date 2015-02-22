/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBinaryShapeBasedSuperSamplingFilter_txx
#define __itkBinaryShapeBasedSuperSamplingFilter_txx

#include "itkBinaryShapeBasedSuperSamplingFilter.h"

#include <itkCastImageFilter.h>
#include <itkExceptionObject.h>
#include <itkImageDuplicator.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>

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
BinaryShapeBasedSuperSamplingFilter<TInputImage, TOutputImage>
::BinaryShapeBasedSuperSamplingFilter()
{

}


/* ---------------------------------------------------------------------
   SmoothDistanceMap()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
typename BinaryShapeBasedSuperSamplingFilter<TInputImage, TOutputImage>::FloatImagePointer
BinaryShapeBasedSuperSamplingFilter<TInputImage, TOutputImage>
::SmoothDistanceMap( unsigned int idim, FloatImagePointer image )
{
  
  // Sampling

  if ( this->m_SamplingFactors[idim] < 1 )
  {
    typename FloatImageType::SpacingType spacing = image->GetSpacing();

    double mmStdDev;  // The standard deviation of the Gaussian to use
    double mmScale;   // The scale in mm of the smoothing

    mmStdDev = spacing[idim]/(2.*sqrt(3.)*this->m_SamplingFactors[idim]);
    mmScale = mmStdDev*mmStdDev/2.;

    if ( this->GetVerbose() ) 
    {
      std::cout << "Dimension: " << idim 
                << "  Spacing: " << spacing[idim]
                << "mm, sampling: " << this->m_SamplingFactors[idim]
                << ", Gaussian standard deviation: " << mmStdDev << "mm"
                << " scale: " << mmScale << "mm" << std::endl; 
    }

    typedef LewisGriffinRecursiveGaussianImageFilter<FloatImageType, FloatImageType> SmootherType;

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
BinaryShapeBasedSuperSamplingFilter<TInputImage, TOutputImage>
::GenerateData()
{
  unsigned int idim;

  for ( idim=0; idim<3; idim++ )
  {
    if ( this->m_SamplingFactors[idim] > 1. )
    {
      itkExceptionMacro( << "Super-sampling factors must be less than or equal to one."
                         << std::endl );
    }

    if ( this->m_SamplingFactors[idim] <= 0. )
    {
      itkExceptionMacro( << "Super-sampling factors greater than zero." 
                         << std::endl );
    }
  }

  // Get the input and output pointers

  InputImageConstPointer  imInput = this->GetInput();


  // Allocate memory for the output

  OutputImagePointer imOutput = this->GetOutput( );

  imOutput->SetBufferedRegion( imOutput->GetRequestedRegion() );
  imOutput->Allocate();
  imOutput->FillBuffer( 0 );


  // Compute a distance map at low resolution

  typedef SignedMaurerDistanceMapImageFilter<InputImageType, FloatImageType> InputDistanceMapFilterType;

  typename InputDistanceMapFilterType::Pointer inputDistFilter = InputDistanceMapFilterType::New();

  inputDistFilter->SetInput( imInput );
  inputDistFilter->SetUseImageSpacing(true);

  inputDistFilter->Update();

  
  // Resample the distance map at the higher resolution

  typedef ResampleImageFilter<FloatImageType, FloatImageType> ResampleType;

  typename ResampleType::Pointer resampleFilter = ResampleType::New();

  switch ( this->m_Interpolation )
  {
  case NEAREST: 
  { 
    std::cout << "Interpolating distance map with nearest neighbour interpolation." << std::endl;
   
    typedef itk::NearestNeighborInterpolateImageFunction< FloatImageType, double> NearestNeighbourInterpolatorType;

    typename NearestNeighbourInterpolatorType::Pointer interpolator 
      = NearestNeighbourInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  case LINEAR: 
  {
    std::cout << "Interpolating distance map with linear interpolation." << std::endl;

    typedef itk::LinearInterpolateImageFunction< FloatImageType, double >         LinearInterpolatorType;

    typename LinearInterpolatorType::Pointer interpolator 
      = LinearInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  case BSPLINE: 
  {
    std::cout << "Interpolating distance map with b-spline interpolation." << std::endl;

    typedef itk::BSplineInterpolateImageFunction< FloatImageType, double >        BSplineInterpolatorType;

    typename BSplineInterpolatorType::Pointer interpolator 
      = BSplineInterpolatorType::New();

    resampleFilter->SetInterpolator( interpolator );
    break;
  }

  case SINC: 
  {
    std::cout << "Interpolating distance map with sinc interpolation." << std::endl;

    typedef itk::ConstantBoundaryCondition< FloatImageType > BoundaryConditionType;
    const static unsigned int WindowRadius = 5;
    typedef itk::Function::WelchWindowFunction<WindowRadius>  WindowFunctionType;
    typedef itk::WindowedSincInterpolateImageFunction< FloatImageType, 
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

  resampleFilter->SetInput( inputDistFilter->GetOutput() );

  typedef itk::IdentityTransform<double,OutputImageType::ImageDimension> IdentityTransformType;

  typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();

  resampleFilter->SetOutputParametersFromImage( imOutput );
  resampleFilter->SetTransform(identityTransform);

  resampleFilter->Modified();
  resampleFilter->UpdateLargestPossibleRegion();

  
  // Smooth the distance map

  FloatImagePointer imMaskDistance = resampleFilter->GetOutput();;

  for ( idim = 0; idim < ImageDimension; idim++ )
  {
    imMaskDistance = SmoothDistanceMap( idim, imMaskDistance );
  }


  // Perform the shape based interpolation by thresholding the
  // distance of non-original voxels from the original set mask voxels

  typename OutputImageType::SpacingType spacing = imOutput->GetSpacing();

  float threshold = 0.;
  for (unsigned int idim=0; idim < OutputImageType::ImageDimension; idim++ ) 
  {
    threshold += spacing[idim]*spacing[idim];
  }
  threshold = vcl_sqrt( threshold );

  itk::ImageRegionIterator< FloatImageType > 
    itMaskDistance( imMaskDistance, imMaskDistance->GetLargestPossibleRegion() );

  for ( itMaskDistance.GoToBegin(); 
        ! itMaskDistance.IsAtEnd(); 
        ++itMaskDistance )
  {
    if ( itMaskDistance.Get() <= threshold*0.6666 )
    {
      itMaskDistance.Set( 255. );
    }
    else
    {
      itMaskDistance.Set( 0. );
    }
  }


  // Create caster for output

  typedef CastImageFilter< FloatImageType, OutputImageType > CasterType;

  typename CasterType::Pointer caster = CasterType::New();

  caster->SetInput( imMaskDistance );
  caster->Update();

  imOutput = caster->GetOutput();
  imOutput->DisconnectPipeline();

  this->GraftOutput( imOutput );
}


/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
BinaryShapeBasedSuperSamplingFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


} // namespace itk

#endif
