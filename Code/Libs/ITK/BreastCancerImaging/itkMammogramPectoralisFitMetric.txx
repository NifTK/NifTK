/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramPectoralisFitMetric_txx
#define __itkMammogramPectoralisFitMetric_txx


#include "itkMammogramPectoralisFitMetric.h"

#include <itkWriteImage.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>

#include <vnl/vnl_math.h>

namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramPectoralisFitMetric<TInputImage>
::MammogramPectoralisFitMetric()
{
  m_InputImage = 0;
  m_Mask = 0;
  m_ImTemplate = 0;
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramPectoralisFitMetric<TInputImage>
::~MammogramPectoralisFitMetric()
{
}


/* -----------------------------------------------------------------------
   SetInputImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramPectoralisFitMetric<TInputImage>
::SetInputImage( const InputImageType *imInput )
{
  m_InputImage = const_cast< InputImageType *>( imInput );

  // Allocate the template image

  m_ImRegion  = m_InputImage->GetLargestPossibleRegion();
  m_ImSpacing = m_InputImage->GetSpacing();
  m_ImOrigin  = m_InputImage->GetOrigin();

  m_ImSize    = m_ImRegion.GetSize();

  m_ImSizeInMM[0] = m_ImSize[0]*m_ImSpacing[0];
  m_ImSizeInMM[1] = m_ImSize[1]*m_ImSpacing[1];

  m_ImTemplate = TemplateImageType::New();

  m_ImTemplate->SetRegions( m_ImRegion );
  m_ImTemplate->SetSpacing( m_ImSpacing );
  m_ImTemplate->SetOrigin(  m_ImOrigin );

  m_ImTemplate->Allocate( );
  m_ImTemplate->FillBuffer( 0 );

  this->Modified();
}


/* -----------------------------------------------------------------------
   SetMask()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramPectoralisFitMetric<TInputImage>
::SetMask( const MaskImageType *imMask )
{
  m_Mask = const_cast< MaskImageType *>( imMask );

  MaskImageRegionType regionMask = m_Mask->GetLargestPossibleRegion();

  MaskLineIteratorType itMaskLinear( m_Mask, regionMask );

  bool flgPixelFound;
  MaskImageIndexType maskIndex, firstMaskPixel, lastMaskPixel;

  itMaskLinear.SetDirection( 1 ); // First 'x' coord
  itMaskLinear.GoToBegin();

  flgPixelFound = false;
  while ( ! itMaskLinear.IsAtEnd() )
  {
    itMaskLinear.GoToBeginOfLine();

    while ( ! itMaskLinear.IsAtEndOfLine() )
    {
      if ( itMaskLinear.Get() )
      {
        flgPixelFound = true;
        break;
      }
      ++itMaskLinear;
    }
    if ( flgPixelFound )
    {
      break;
    }
    itMaskLinear.NextLine();
  }
  maskIndex = itMaskLinear.GetIndex();
  firstMaskPixel[0] = maskIndex[0];

  itMaskLinear.SetDirection( 0 ); // First 'y' coord
  itMaskLinear.GoToBegin();

  flgPixelFound = false;
  while ( ! itMaskLinear.IsAtEnd() )
  {
    itMaskLinear.GoToBeginOfLine();

    while ( ! itMaskLinear.IsAtEndOfLine() )
    {
      if ( itMaskLinear.Get() )
      {
        flgPixelFound = true;
        break;
      }
      ++itMaskLinear;
    }
    if ( flgPixelFound )
    {
      break;
    }
    itMaskLinear.NextLine();
  }
  maskIndex = itMaskLinear.GetIndex();
  firstMaskPixel[1] = maskIndex[1];


  itMaskLinear.SetDirection( 1 ); // Last 'x' coord
  itMaskLinear.GoToReverseBegin();

  flgPixelFound = false;
  while ( ! itMaskLinear.IsAtReverseEnd() )
  {
    itMaskLinear.GoToBeginOfLine();

    while ( ! itMaskLinear.IsAtEndOfLine() )
    {
      if ( itMaskLinear.Get() )
      {
        flgPixelFound = true;
        break;
      }
      ++itMaskLinear;
    }
    if ( flgPixelFound )
    {
      break;
    }
    itMaskLinear.PreviousLine();
  }
  maskIndex = itMaskLinear.GetIndex();
  lastMaskPixel[0] = maskIndex[0];

  itMaskLinear.SetDirection( 0 ); // Last 'y' coord
  itMaskLinear.GoToReverseBegin();

  flgPixelFound = false;
  while ( ! itMaskLinear.IsAtReverseEnd() )
  {
    itMaskLinear.GoToBeginOfLine();

    while ( ! itMaskLinear.IsAtEndOfLine() )
    {
      if ( itMaskLinear.Get() )
      {
        flgPixelFound = true;
        break;
      }
      ++itMaskLinear;
    }
    if ( flgPixelFound )
    {
      break;
    }
    itMaskLinear.PreviousLine();
  }
  maskIndex = itMaskLinear.GetIndex();
  lastMaskPixel[1] = maskIndex[1];

  MaskImageSizeType size = regionMask.GetSize();

  m_MaskRegion.SetIndex( firstMaskPixel );

  size[0] = lastMaskPixel[0] - firstMaskPixel[0] + 1;
  size[1] = lastMaskPixel[1] - firstMaskPixel[1] + 1;

  m_MaskRegion.SetSize( size );  

  if ( this->GetDebug() )
  {
    std::cout << "First mask pixel: " << firstMaskPixel << std::endl
              << "Last mask pixel:  " << lastMaskPixel << std::endl
              << "Mask region: " << m_MaskRegion << std::endl;
  }

  this->Modified();
}


/* -----------------------------------------------------------------------
   ClearTemplate()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void
MammogramPectoralisFitMetric<TInputImage>
::ClearTemplate( void )
{
  if ( ! m_ImTemplate )
  {
    itkExceptionMacro( << "ERROR: Template image not allocated." );
    return;
  }

  m_ImTemplate->FillBuffer( 0 );
}


/* -----------------------------------------------------------------------
   GetParameters()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void
MammogramPectoralisFitMetric<TInputImage>
::GetParameters( const InputImagePointType &pecInterceptInMM,
                 ParametersType &parameters )
{
  double w, h;

  w = pecInterceptInMM[0];
  h = pecInterceptInMM[1];
  
  parameters[0] = h;            // The Gompertz curve pectoral profile
  parameters[1] = -4.;
  parameters[2] = log( log( 0.9 )/parameters[1] )/w;

  parameters[3] = 0.;           // Translation in x
  parameters[4] = 0.;           // Translation in y

  parameters[5] = 0.;           // The rotation theta

  parameters[6] = 2.;           // 1/(d + x) : the intensity profile

  parameters[7] = 0.5;          // The slope in 'y'
}


/* -----------------------------------------------------------------------
   GradientAtMidpoint()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
double 
MammogramPectoralisFitMetric<TInputImage>
::GradientAtMidpoint( const ParametersType &parameters ) const
{
  double a = parameters[0];
  double b = parameters[1];
  double c = parameters[2];

  // Compute the 'x' coordinate of y = a/2

  double x;

  x = log( log(0.5)/b )/c;
  
  // Compute the derivative

  double exp_cx = exp( c*x );

  return a*b*c*exp( b*exp_cx )*exp_cx;
}


/* -----------------------------------------------------------------------
   GenerateTemplate()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramPectoralisFitMetric<TInputImage>
::GenerateTemplate( const ParametersType &parameters,
                    double &tMean, double &tStdDev, double &nInside, double &nPixels,
                    TemplateImageRegionType &templateRegion )
{
  if ( ! m_InputImage )
  {
    itkExceptionMacro( << "ERROR: Input image not set." );
    return;
  }

  tMean = 0.;
  tStdDev = 1.;

  nPixels = 0;
  nInside = 0;

  double nOutside = 0;

  double value;
  double x, y;

  double a = parameters[0];
  double b = parameters[1];
  double c = parameters[2];

  double tx    = -parameters[3];
  double ty    = -parameters[4];

  double theta = -parameters[5]*vnl_math::pi/180.0; // Convert to radians

  // The coefficient for the muscle profile should be have a minimum
  // of 1 but be asymptotic to y = x i.e. hyperbolic

  double profile =  sqrt ( 1 + parameters[6]*parameters[6] );
  
  double sinTheta = sin( theta );
  double cosTheta = cos( theta );

  double ay = fabs( parameters[7] );    // The slope in 'y'

  if ( ay > 1. )
  {
    ay = 1.;
  }


  InputImageIndexType index;
  InputImageIndexType maxIndex;
  InputImagePointType point;

  TemplateImageSizeType   templateSize;

  MaskIteratorType *itMask = 0;

  if ( m_Mask )
  {
    itMask = new MaskIteratorType( m_Mask, m_MaskRegion );
    itMask->GoToBegin();

    templateRegion = m_MaskRegion;
  }
  else
  {
    templateRegion = m_ImTemplate->GetLargestPossibleRegion();
  }

  TemplateIteratorWithIndexType itTemplate( m_ImTemplate, templateRegion );

  maxIndex = templateRegion.GetIndex();

  for ( itTemplate.GoToBegin();
        ! itTemplate.IsAtEnd();
        ++itTemplate )
  {
    if ( (! itMask) || itMask->Get() )
    {

      index = itTemplate.GetIndex();

      m_InputImage->TransformIndexToPhysicalPoint( index, point );      
    
      x = point[0]*cosTheta - point[1]*sinTheta + tx;
      y = point[0]*sinTheta + point[1]*cosTheta + ty;

      if ( (0.8*a - y) > a*exp( b*exp( c*x ) ) )
      {
        itTemplate.Set( 1. );

        if ( index[0] > maxIndex[0] ) maxIndex[0] = index[0];
        if ( index[1] > maxIndex[1] ) maxIndex[1] = index[1];
      }
      else
      {
        itTemplate.Set( 0. );
      }
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }

  double yStart  = static_cast<double>( templateRegion.GetIndex()[1] );
  double yLength = static_cast<double>( maxIndex[1] ) - yStart;

  // Compute the distance transform to simulate the muscle profile via d/(1 + d)

  typedef typename itk::SignedMaurerDistanceMapImageFilter< TemplateImageType, 
                                                            TemplateImageType> DistanceTransformType;
  
  typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

  distanceTransform->SetInput( m_ImTemplate );
  distanceTransform->SetInsideIsPositive( true );
  distanceTransform->UseImageSpacingOn();
  distanceTransform->SquaredDistanceOff();

  distanceTransform->UpdateLargestPossibleRegion();

  TemplateImagePointer imDistTrans = distanceTransform->GetOutput();

  TemplateIteratorType itDistTrans( imDistTrans, templateRegion );

  if ( m_Mask )
  {
    itMask->GoToBegin();
  }

  nInside = 0;
  nOutside = 0;

  for ( itTemplate.GoToBegin(), itDistTrans.GoToBegin();
        ! itTemplate.IsAtEnd();
        ++itTemplate, ++itDistTrans )
  {
    if ( (! itMask) || itMask->Get() ) 
    {
      index = itTemplate.GetIndex();

      if ( itTemplate.Get() > 0 )
      {
        y = itTemplate.GetIndex()[1];

        itTemplate.Set( (1. - ay*(y - yStart)/yLength)
                        *itDistTrans.Get()/( profile + itDistTrans.Get()) );

        if ( index[0] > maxIndex[0] ) maxIndex[0] = index[0];
        if ( index[1] > maxIndex[1] ) maxIndex[1] = index[1];

        nInside++;
      }
      else if ( itDistTrans.Get() > -10. )
      {
        itTemplate.Set( -0.0001 ); // Close to zero but non-zero

        if ( index[0] > maxIndex[0] ) maxIndex[0] = index[0];
        if ( index[1] > maxIndex[1] ) maxIndex[1] = index[1];

        nOutside++;
      }
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }
  
  nPixels = nInside + nOutside;

  if ( nPixels == 0 )
  {
    return;
  }


  // Compute the mean and standard deviation

  if ( m_Mask )
  {
    itMask->GoToBegin();
  }

  // Compute the mean image intensity for this region
  
  tMean = 0;
  
  for ( itTemplate.GoToBegin();
        ! itTemplate.IsAtEnd();
        ++itTemplate )
  {
    if ( ( (! itMask) || itMask->Get() ) && ( itTemplate.Get() ) )
    {
      tMean += itTemplate.Get();
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }
  
  tMean /= nPixels;

  // Compute the standard deviation for this region

  if ( m_Mask )
  {
    itMask->GoToBegin();
  }

  tStdDev = 0;
       
  for ( itTemplate.GoToBegin();
        ! itTemplate.IsAtEnd();
        ++itTemplate )
  {
    if ( ( (! itMask) || itMask->Get() ) && ( itTemplate.Get() ) )
    {
      value = static_cast<double>( itTemplate.Get() ) - tMean;
      tStdDev += value*value;
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }

  if ( tStdDev == 0 )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: Template standard deviation is zero, skipping: " 
                << parameters << std::endl;
    }
  }
  else
  {
    tStdDev = sqrt( tStdDev/nPixels );
  }

  if ( itMask )
  {
    delete itMask;
  }

  templateSize[0] = maxIndex[0] - templateRegion.GetIndex()[0] + 1;
  templateSize[1] = maxIndex[1] - templateRegion.GetIndex()[1] + 1;

  templateRegion.SetSize( templateSize );
  

  if ( this->GetDebug() )
  {
    std::cout << "Mean : " << tMean << " = " << "(" << nInside << " - " 
              << nOutside << ")/" << nPixels << std::endl
              << "Template region: " << maxIndex << std::endl;
  }
}


/* -----------------------------------------------------------------------
   GetValue()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
typename MammogramPectoralisFitMetric<TInputImage>::MeasureType 
MammogramPectoralisFitMetric<TInputImage>
::GetValue( const  InputImagePointType &pecInterceptInMM )
{
  ParametersType parameters;

  parameters.SetSize( ParametricSpaceDimension );

  GetParameters( pecInterceptInMM, parameters );

  MeasureType measure = GetValue( parameters );

#if 0
  char filename[256];
  sprintf( filename, "Template_%05.2f_%05.1fx%05.1f.nii.gz",
           measure, pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TemplateImageType >( filename, "test template image", m_ImTemplate ); 
#endif

  return measure;
}


/* -----------------------------------------------------------------------
   GetValue()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
typename MammogramPectoralisFitMetric<TInputImage>::MeasureType 
MammogramPectoralisFitMetric<TInputImage>
::GetValue( const ParametersType &parameters ) const
{
  if ( ! m_InputImage )
  {
    itkExceptionMacro( << "ERROR: Input image not set." );
    return -1.;
  }

  if ( m_Mask )
  {
    MaskImageSizeType maskSize = m_Mask->GetLargestPossibleRegion().GetSize();
    
    if ( ( maskSize[0] != m_ImSize[0] ) || ( maskSize[1] != m_ImSize[1] ) )
    {
      itkExceptionMacro( << "ERROR: Mask dimensions, " << maskSize 
                         << ", do not match input image, " << m_ImSize );
      return -1.;
    }
  } 

  // Only permit muscle boundaries with slopes greater than 45 degrees

  if ( this->GradientAtMidpoint( parameters ) < 1. )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: Slope of muscle boundary is less than 1, skipping: " 
                << parameters << std::endl;
    }
    return -1.;
  }
    

  double value;
  double nPixels, nPecPixels;
  double imMean, imStdDev;
  double tMean, tStdDev;

  MeasureType ncc = 0.;


  // Create the template

  TemplateImageRegionType templateRegion;

  const_cast< MammogramPectoralisFitMetric<TInputImage>* >(this)->GenerateTemplate( parameters, 
                                                                                    tMean, tStdDev, 
                                                                                    nPecPixels, nPixels,
                                                                                    templateRegion );
  

  if ( nPixels == 0 )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: No pixels in template, skipping: " 
                << parameters << std::endl;
    }
    return -1.;
  }

  if ( ( tMean < -0.5 ) || ( tMean > 0.5 ) )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: Insufficient pixels in template (mean = " 
                << tMean << ", nPixels = " << nPixels << "), skipping: " << parameters << std::endl;
    }
    return -1.;
  }


  if ( tStdDev == 0 )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: Template standard deviation is zero, skipping: " 
                << parameters << std::endl;
    }
    return -1.;
  }


  // Create the image region iterator

  MaskIteratorType *itMask = 0;

  if ( m_Mask )
  {
    itMask = new MaskIteratorType( m_Mask, m_MaskRegion );
    itMask->GoToBegin();
  }

  IteratorConstType itPecRegion(   m_InputImage, templateRegion );
  TemplateIteratorType itTemplate( m_ImTemplate, templateRegion );

  // Compute the mean image intensity for this region
  
  imMean = 0;
  
  for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
        ! itPecRegion.IsAtEnd();
        ++itPecRegion, ++itTemplate )
  {
    if ( (! itMask) || itMask->Get() )
    {
      if ( itTemplate.Get() )
      {
        imMean += itPecRegion.Get();
      }
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }
  
  imMean /= nPixels;

  // Compute the standard deviation for this region

  if ( m_Mask )
  {
    itMask->GoToBegin();
  }

  imStdDev = 0;
       
  for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
        ! itPecRegion.IsAtEnd();
        ++itPecRegion, ++itTemplate )
  {
    if ( (! itMask) || itMask->Get() )
    {
      if ( itTemplate.Get() )
      {
        value = static_cast<double>( itPecRegion.Get() ) - imMean;
        imStdDev += value*value;
      }
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }

  if ( imStdDev == 0 )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: Image standard deviation is zero, skipping: " 
                << parameters << std::endl;
    }
    return -1.;
  }
  
  imStdDev = sqrt( imStdDev/nPixels );

  // Compute the cross correlation

  if ( m_Mask )
  {
    itMask->GoToBegin();
  }

  ncc = 0;

  for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
        ! itPecRegion.IsAtEnd();
        ++itPecRegion, ++itTemplate )
  {
    if ( (! itMask) || itMask->Get() )
    {
      if ( itTemplate.Get() )
      {
        ncc += 
          ( static_cast<double>( itPecRegion.Get() ) - imMean )
          *( static_cast<double>( itTemplate.Get() ) - tMean );
      }
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }

  if ( itMask )
  {
    delete itMask;
  }

  ncc /= nPixels*imStdDev*tStdDev;

  //ncc *= sqrt( nPecPixels );

  if ( 0 && this->GetDebug() )
  {
    std::cout << "NCC: "           << std::setw(12) << ncc 
              << " Parameters: "   << std::setw(12) << parameters
              << " Region start: " << std::setw(12) << templateRegion.GetIndex() 
              << ", size: "        << std::setw(12) << templateRegion.GetSize() << std::endl;
  }

  return ncc;
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage>
void
MammogramPectoralisFitMetric<TInputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
