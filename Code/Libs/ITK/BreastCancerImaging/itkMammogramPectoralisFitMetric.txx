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

namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramPectoralisFitMetric<TInputImage>
::MammogramPectoralisFitMetric()
{
  m_BreastSide = LeftOrRightSideCalculatorType::UNKNOWN_BREAST_SIDE;

  m_InputImage = 0;
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

  // Calculate the breast side

  typename LeftOrRightSideCalculatorType::Pointer 
    sideCalculator = LeftOrRightSideCalculatorType::New();

  sideCalculator->SetImage( m_InputImage );

  sideCalculator->Compute();

  m_BreastSide = sideCalculator->GetBreastSide();

  // Allocate the template image

  m_ImRegion  = m_InputImage->GetLargestPossibleRegion();
  m_ImSpacing = m_InputImage->GetSpacing();
  m_ImOrigin  = m_InputImage->GetOrigin();

  m_ImSize    = m_ImRegion.GetSize();

  m_ImSizeInMM[0] = m_ImSize[0]*m_ImSpacing[0];
  m_ImSizeInMM[1] = m_ImSize[1]*m_ImSpacing[1];

  m_ImTemplate = InputImageType::New();

  m_ImTemplate->SetRegions( m_ImRegion );
  m_ImTemplate->SetSpacing( m_ImSpacing );
  m_ImTemplate->SetOrigin(  m_ImOrigin );

  m_ImTemplate->Allocate( );
  m_ImTemplate->FillBuffer( 0 );

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
   GetRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void
MammogramPectoralisFitMetric<TInputImage>
::GetRegion( const ParametersType &parameters,
             InputImageRegionType &region ) const
{
  if ( ! m_InputImage )
  {
    itkExceptionMacro( << "ERROR: Input image not set." );
    return;
  }

  unsigned int i;

  double a = parameters[0];
  double b = parameters[1];
  double c = parameters[2];

  InputImagePointType intercepts;

  if ( m_BreastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
  {
    intercepts[0] = log( log( 0.9 )/b )/c;
    intercepts[1] = a;
  }
  else
  {
    intercepts[0] = m_ImSizeInMM[0] - log( log( 0.9 )/b )/c;
    intercepts[1] = a;
  }

  InputImageIndexType index;
  
  m_InputImage->TransformPhysicalPointToIndex( intercepts, index );

  InputImageSizeType size;
  InputImageIndexType start;
  
  if ( m_BreastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
  {
    size[0] = index[0];
    size[1] = index[1];

    start[0] = 0;
    start[1] = 0;
  }
  else
  {
    size[0] = m_ImSize[0] - index[0] - 1;
    size[1] = index[1];

    start[0] = index[0];
    start[1] = 0;
  }

  for ( i=0; i<2; i++ )
  {
    if ( size[i] < 3 ) 
    {
      size[i] = 3;
    }
    else if ( size[i] >= m_ImSize[i] ) 
    {
      size[i] = m_ImSize[i];
    }

    if ( start[i] >= m_ImSize[i] - size[i] ) 
    {
      start[i] = m_ImSize[i] - size[i] - 1;
    }

    if ( start[i] < 0 ) 
    {
      start[i] = 0;
    }
  }

  region.SetSize( size );
  region.SetIndex( start );
}


/* -----------------------------------------------------------------------
   GetDefaultParameters()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void
MammogramPectoralisFitMetric<TInputImage>
::GetParameters( const InputImagePointType &pecInterceptInMM,
                 ParametersType &parameters )
{
  double w, h;

  if ( m_BreastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
  {
    w = pecInterceptInMM[0];
    h = pecInterceptInMM[1];
  }
  else if ( m_BreastSide == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
  {
    w = m_ImSizeInMM[0] - pecInterceptInMM[0];
    h = pecInterceptInMM[1];
  }
  else
  {
    itkExceptionMacro( << "ERROR: Breast side not defined, input image unset?" );
    return;
  }
  
  parameters[0] = h;
  parameters[1] = -4.;
  parameters[2] = log( log( 0.9 )/parameters[1] )/w;
}


/* -----------------------------------------------------------------------
   GenerateTemplate()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramPectoralisFitMetric<TInputImage>
::GenerateTemplate( const ParametersType &parameters,
                    double &tMean, double &tStdDev, double &nPixels ) const
{
  if ( ! m_InputImage )
  {
    itkExceptionMacro( << "ERROR: Input image not set." );
    return;
  }

  unsigned int nInside = 0, nOutside = 0;

  double x, y;

  InputImageRegionType region;
                    
  GetRegion( parameters, region );

  double a = parameters[0];
  double b = parameters[1];
  double c = parameters[2];

  InputImageIndexType index;
  InputImageIndexType start;

  InputImagePointType point;
  InputImagePointType ptStart;

  start = region.GetIndex();

  m_InputImage->TransformIndexToPhysicalPoint( start, ptStart );

  tMean = 0.;
  tStdDev = 1.;

  nPixels = 0;

  IteratorWithIndexType itTemplateWithIndex( m_ImTemplate, region );

  for ( itTemplateWithIndex.GoToBegin();
        ! itTemplateWithIndex.IsAtEnd();
        ++itTemplateWithIndex )
  {
    index = itTemplateWithIndex.GetIndex();

    m_InputImage->TransformIndexToPhysicalPoint( index, point );      
    
    if ( m_BreastSide == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
    {
      x = static_cast<double>( m_ImSizeInMM[0] - point[0] );
      y = static_cast<double>( point[1] );        
    }
    else
    {
      x = static_cast<double>( point[0] );
      y = static_cast<double>( point[1] );        
    }

    if ( (0.8*a - y) > a*exp( b*exp( c*x ) ) )
    {
      itTemplateWithIndex.Set( 1. );
      nInside++;
    }
    else if ( (a - y) > a*exp( 1.5*b*exp( c*x ) ) )
    {
      itTemplateWithIndex.Set( -1. );
      nOutside++;
    }
    else
    {
      itTemplateWithIndex.Set( 0. );
    }
  }

  nPixels = nInside + nOutside;

  if ( nPixels == 0 )
  {
    return;
  }

  tMean = ( nInside - nOutside )/nPixels;

  tStdDev = sqrt( (  nInside*(  1 - tMean)*( 1 - tMean)
                   + nOutside*(-1 - tMean)*(-1 - tMean) )/nPixels );
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

  double value;
  double nPixels;
  double imMean, imStdDev;
  double tMean, tStdDev;

  MeasureType ncc = 0.;


  // Create the template

  typename TInputImage::RegionType pecRegion;

  GenerateTemplate( parameters, tMean, tStdDev, nPixels );

  if ( nPixels == 0 )
  {
    if ( this->GetDebug() )
    {
      std::cout << "WARNING: No pixels in template, skipping: " 
                << parameters << std::endl;
    }
    return -1.;
  }

  GetRegion( parameters, pecRegion );


  // Create the image region iterator

  IteratorConstType itPecRegion( m_InputImage, pecRegion );
  IteratorType itTemplate( m_ImTemplate, pecRegion );

  // Compute the mean image intensity for this region
  
  imMean = 0;
  
  for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
        ! itPecRegion.IsAtEnd();
        ++itPecRegion, ++itTemplate )
  {
    if ( itTemplate.Get() )
    {
      imMean += itPecRegion.Get();
    }
  }
  
  imMean /= nPixels;

  // Compute the standard deviation for this region

  imStdDev = 0;
       
  for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
        ! itPecRegion.IsAtEnd();
        ++itPecRegion, ++itTemplate )
  {
    if ( itTemplate.Get() )
    {
      value = static_cast<double>( itPecRegion.Get() ) - imMean;
      imStdDev += value*value;
    }
  }

  imStdDev = sqrt( imStdDev/nPixels );

  // Compute the cross correlation

  ncc = 0;

  for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
        ! itPecRegion.IsAtEnd();
        ++itPecRegion, ++itTemplate )
  {
    if ( itTemplate.Get() )
    {
      ncc += 
        ( static_cast<double>( itPecRegion.Get() ) - imMean )
        *( static_cast<double>( itTemplate.Get() ) - tMean )
        / ( imStdDev*tStdDev);
    }
  }

  ncc /= nPixels;

  if ( 0 && this->GetDebug() )
  {
    std::cout << "NCC: " << std::setw(12) << ncc 
              << " Parameters: " << std::setw(12) << parameters
              << " Region start: " << std::setw(12) << pecRegion.GetIndex() 
              << ", size: " << std::setw(12) << pecRegion.GetSize() << std::endl;
  }

  return ncc;
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

  return GetValue( parameters );
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
