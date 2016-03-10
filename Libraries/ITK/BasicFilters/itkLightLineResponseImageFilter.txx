/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkLightLineResponseImageFilter_txx
#define __itkLightLineResponseImageFilter_txx

#include "itkLightLineResponseImageFilter.h"
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageFileWriter.h>
#include <itkImageDuplicator.h>
#include <itkProgressReporter.h>
#include <niftkConversionUtils.h>

#include <vnl/vnl_double_2x2.h>

#include <itkUCLMacro.h>


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
LightLineResponseImageFilter<TInputImage,TOutputImage>
::LightLineResponseImageFilter()
{

  // Multi-threaded execution is enabled by default
  m_FlagMultiThreadedExecution = true;

  m_Sigma = 1.;

  m_Epsilon = 1.0e-05;
  m_NormalizeAcrossScale = false;

  m_FlagLocalOrientationSet = false;

  m_NumberOfOrientations = 8;

  m_FlagOriginSet = false;

  flipHorizontally = 1.;
  flipVertically = 1.;

  m_Origin[0] = 0.;
  m_Origin[1] = 0.;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_Mask = 0;
  m_MaskFilter = 0;
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
LightLineResponseImageFilter<TInputImage,TOutputImage>
::~LightLineResponseImageFilter()
{
}

/* -----------------------------------------------------------------------
   SetSigma()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::SetSigma( RealType sigma )
{
  m_Sigma = sigma;
  this->Modified();
}


/* -----------------------------------------------------------------------
   SetNormalizeAcrossScale()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::SetNormalizeAcrossScale( bool normalize )
{
  m_NormalizeAcrossScale = normalize;

  this->Modified();
}

/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out) 
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
}


/* -----------------------------------------------------------------------
   GetDerivative()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
typename LightLineResponseImageFilter<TInputImage,TOutputImage>::RealImagePointer 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::GetDerivative( DerivativeFilterOrderEnumTypeX xOrder,
		 DerivativeFilterOrderEnumTypeY yOrder )
{
  DerivativeFilterPointerX derivativeFilterX;
  DerivativeFilterPointerY derivativeFilterY;

  derivativeFilterX = DerivativeFilterTypeX::New();
  derivativeFilterY = DerivativeFilterTypeY::New();

  derivativeFilterX->SetSingleThreadedExecution();
  derivativeFilterY->SetSingleThreadedExecution();

  derivativeFilterX->SetSigma( m_Sigma );
  derivativeFilterY->SetSigma( m_Sigma );

  derivativeFilterX->SetInput( this->GetInput() );
  derivativeFilterY->SetInput( derivativeFilterX->GetOutput() );

  derivativeFilterX->SetDirection( 0 );
  derivativeFilterY->SetDirection( 1 );

  derivativeFilterX->SetOrder( xOrder );
  derivativeFilterY->SetOrder( yOrder );

  if ( m_Mask ) {
    derivativeFilterX->SetMask( m_Mask );
    derivativeFilterY->SetMask( m_Mask );
  }

  if ( this->GetDebug() )
  {
    derivativeFilterX->SetDebug( true );
    derivativeFilterY->SetDebug( true );
  }

  derivativeFilterY->Update(); 

  return derivativeFilterY->GetOutput();
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData(void)
{
  typedef itk::ImageFileWriter< RealImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();

  // S11

  niftkitkDebugMacro(<< "Computing S11");

  m_S11 = GetDerivative(DerivativeFilterTypeX::FirstOrder,
			DerivativeFilterTypeY::FirstOrder );

  // S20

  niftkitkDebugMacro(<< "Computing S02");

  m_S20 = GetDerivative(DerivativeFilterTypeX::SecondOrder,
			DerivativeFilterTypeY::ZeroOrder );

  // S02

  niftkitkDebugMacro(<< "Computing S20");

  m_S02 = GetDerivative(DerivativeFilterTypeX::ZeroOrder,
			DerivativeFilterTypeY::SecondOrder );


  // Allocate the filter response images

  typename RealImageType::RegionType region = this->GetInput()->GetLargestPossibleRegion();

  m_Orientation = RealImageType::New();

  m_Orientation->SetRegions( region );
  m_Orientation->SetSpacing( this->GetInput()->GetSpacing() );
  m_Orientation->SetOrigin( this->GetInput()->GetOrigin() );
  m_Orientation->Allocate( );
  m_Orientation->FillBuffer( 0. );
}


/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */
template< typename TInputImage, typename TOutputImage >
void
LightLineResponseImageFilter< TInputImage, TOutputImage >
::GenerateInputRequestedRegion()
throw( InvalidRequestedRegionError )
{
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();

  // This filter needs all of the input
  typename LightLineResponseImageFilter< TInputImage, TOutputImage >
    ::InputImagePointer image = const_cast< InputImageType * >( this->GetInput() );

  if ( image )
  {
    image->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
  }
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  // Perform multi-threaded execution by default

  if (m_FlagMultiThreadedExecution) {
    
    niftkitkDebugMacro( "Multi-threaded basic image features");

    Superclass::GenerateData();
  }

  // Single-threaded execution

  else {
  
    niftkitkDebugMacro( "Single-threaded basic image features");

    this->AllocateOutputs();
    this->BeforeThreadedGenerateData();
  
    // Set up the multithreaded processing
    LightLineResponseThreadStruct str;
    str.Filter = this;
    
    this->GetMultiThreader()->SetNumberOfThreads( 1 );
    this->GetMultiThreader()->SetSingleMethod(this->ThreaderCallback, &str);
    
    // multithread the execution
    this->GetMultiThreader()->SingleMethodExecute();
    
    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    this->AfterThreadedGenerateData();
  }
}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const OutputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
LightLineResponseImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId)
{
  vnl_double_2 *dirn = 0;		// The list of vector orientations
  vnl_double_2 *orient = 0;		// Vector orientations for the current pixel (eg. wrt. to origin)

  niftkitkDebugMacro( << "LightLine Region: " << outputRegionForThread );

  dirn = new vnl_double_2[m_NumberOfOrientations];
  orient = new vnl_double_2[m_NumberOfOrientations];

  unsigned int k, kStart;
  double theta;
  vnl_double_2x2 R;

  dirn[0][0] = 1.; dirn[0][1] = 0.;

  for ( k=1; k<m_NumberOfOrientations; k++ ) {

    theta = k*2.*vnl_math::pi/m_NumberOfOrientations;

    R( 0, 0 ) = vcl_cos( theta ); R( 0, 1 ) =  vcl_sin( theta ); 
    R( 1, 0 ) = vcl_sin( theta ); R( 1, 1 ) = -vcl_cos( theta );

    dirn[k] = R*dirn[0];

    if ( flipHorizontally != 1. ) 
      dirn[k][0] *= flipHorizontally;

    if ( flipVertically != 1. ) 
      dirn[k][1] *= flipVertically;
  }

  for ( k=0; k<m_NumberOfOrientations; k++ ) 
    orient[k] = dirn[k];



  OutputImageIndexType index;

  // Allocate output

  InputImageConstPointer inImage  = this->GetInput();
  OutputImagePointer     outImage = this->GetOutput();

  // Support progress methods/callbacks

  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  // Iterate over pixels in the 2D projection (i.e. output) image

  ImageRegionIterator< OutputImageType > outputIterator
    = ImageRegionIterator< OutputImageType >( outImage, outputRegionForThread );

  ImageRegionIterator< RealImageType > itS11
    = ImageRegionIterator< RealImageType >( m_S11, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS20
    = ImageRegionIterator< RealImageType >( m_S20, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS02
    = ImageRegionIterator< RealImageType >( m_S02, outputRegionForThread );


  ImageRegionIterator< RealImageType > itOrientation
    = ImageRegionIterator< RealImageType >( m_Orientation, outputRegionForThread );


  outputIterator.GoToBegin();

  itS11.GoToBegin();
  itS20.GoToBegin();
  itS02.GoToBegin();

  itOrientation.GoToBegin();

  InputImageSpacingType spacing = inImage->GetSpacing();

  unsigned int idx;

  RealType tmp; 
  RealType maximum; 

  RealType lambda;
  RealType gamma;
  RealType sigma = vcl_sqrt(2.0f)*m_Sigma/sqrt(spacing[0]*spacing[0] + spacing[1]*spacing[1]);
  RealType sigmae = vcl_sqrt(sigma*sigma + 1.0f/9.0f);
  RealType sigmaeSqr = sigmae*sigmae;
  RealType oneSqrtTwo = 1.0f/vcl_sqrt(2.0f);

  for ( ; ! outputIterator.IsAtEnd(); ++outputIterator) {


    lambda = sigmaeSqr*(itS20.Get() + itS02.Get())/2.;
    gamma = sigmaeSqr*vcl_sqrt( ( itS20.Get() - itS02.Get() )*( itS20.Get() - itS02.Get() )/4. + itS11.Get()*itS11.Get() );

    outputIterator.Set( oneSqrtTwo*( gamma - lambda ) );

    
    // Calculate the orientation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    double theta;
    OutputImagePointType vReference;    
    vnl_double_2 vStructure;
	
    vStructure.fill( 0. );

    double S20 = itS20.Get(); 
    double S11 = itS11.Get(); 
    double S02 = itS02.Get();

    double lambda1 = ((S20 + S02) + sqrt((S20 + S02)*(S20 + S02) - 4*(S20*S02 - S11*S11)))/2.;
    double lambda2 = ((S20 + S02) - sqrt((S20 + S02)*(S20 + S02) - 4*(S20*S02 - S11*S11)))/2.;

    if ( fabs( lambda1 ) > fabs( lambda2 ) ) {
      vStructure( 0 ) = S20 - lambda1;
      vStructure( 1 ) = S11;
    }
    else {
      vStructure( 0 ) = S20 - lambda2;
      vStructure( 1 ) = S11;
    }
    
    double mag = sqrt( vStructure( 0 )*vStructure( 0 ) + vStructure( 1 )*vStructure( 1 ) );
    
    vStructure( 0 ) /= mag;
    vStructure( 1 ) /= mag;

  
    // Has a reference orientation been specified

    if (m_FlagLocalOrientationSet) 
    {

      index = outputIterator.GetIndex();
      outImage->TransformIndexToPhysicalPoint( index, vReference );
      
      typename RealImageType::IndexType orientIndex;
      typename RealImageType::PointType orientPoint;
      
      orientPoint = vReference;
      
      if ( m_OrientationInX->TransformPhysicalPointToIndex( orientPoint, orientIndex ) ) {
        
        vReference[0] = m_OrientationInX->GetPixel( orientIndex );
        vReference[1] = m_OrientationInY->GetPixel( orientIndex );
      }
      
      theta = atan2(vReference[1], vReference[0]) - atan2(vStructure(1), vStructure(0));
    }
    
    // Or are we interested in the orientation w.r.t. the origin
    
    else if ( m_FlagOriginSet ) 
    {
      
      index = outputIterator.GetIndex();
      
      outImage->TransformIndexToPhysicalPoint( index, vReference );
      vReference[0] -= m_Origin[0];
      vReference[1] -= m_Origin[1];
      
      theta = atan2(vReference[1], vReference[0]) - atan2(vStructure(1), vStructure(0));
    }
    
    // Otherwise use absolute orientation

    else 
    {
      theta = atan2(vStructure(1), vStructure(0));
    }

    // Ensure angle is between 0 and pi/2 for second order
      
    if ( theta < 0 )
    {
      theta = -theta;  
    }

    if ( theta > vnl_math::pi )
    {
      theta = 2*vnl_math::pi - theta;
    }

    if ( theta > vnl_math::pi/2. )
    {
      theta = vnl_math::pi - theta;
    }

    itOrientation.Set( theta );

    ++itS11;
    ++itS20;
    ++itS02;

    ++itOrientation;
  }

  if (dirn)    delete[] dirn;
  if (orient)  delete[] orient;
}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
LightLineResponseImageFilter<TInputImage,TOutputImage>
::AfterThreadedGenerateData(void)
{
  if ( m_Mask ) {
    
    m_MaskFilter = MaskFilterType::New();

    m_MaskFilter->SetInput1( this->GetOutput() );
    m_MaskFilter->SetInput2( m_Mask );

    m_MaskFilter->Update( );

    this->GraftOutput( m_MaskFilter->GetOutput() );
  }
}


/* -----------------------------------------------------------------------
   WriteDerivativeToFile()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
LightLineResponseImageFilter<TInputImage,TOutputImage>
::WriteDerivativeToFile( int n, std::string filename ) 
{
  typename RealImageType::Pointer inputImage;

  typedef itk::ImageFileWriter< RealImageType > DerivativesWriterType;
  typename DerivativesWriterType::Pointer derivsWriter = DerivativesWriterType::New();

  typedef typename itk::MaskImageFilter< RealImageType, 
                                         MaskImageType, 
                                         RealImageType > DerivativeMaskFilterType;
  typename DerivativeMaskFilterType::Pointer maskFilter;


  switch ( n ) 
    {

    case 3: { inputImage = m_S11; break; }
    case 4: { inputImage = m_S20; break; }
    case 5: { inputImage = m_S02; break; }
      
    default : {
      niftkitkErrorMacro(<< "Derivative number 'n' must satisfy: 0 < n < 6");
      return;
    }
    }


  if ( m_Mask ) {
    
    maskFilter = DerivativeMaskFilterType::New();

    maskFilter->SetInput1( inputImage );
    maskFilter->SetInput2( m_Mask );

    derivsWriter->SetInput( maskFilter->GetOutput() );
  }

  else 
    derivsWriter->SetInput( inputImage );

  std::cout << "Writing: " << filename << std::endl;
  derivsWriter->SetFileName( filename.c_str() );

  try
    {
      derivsWriter->Update();
    }
  catch (itk::ExceptionObject &err)
    {
      std::cerr << "ERROR: Failed to write derivative to file: " << filename << "; " << err << std::endl;
    }
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
LightLineResponseImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (m_FlagMultiThreadedExecution)
    os << indent << "MultiThreadedExecution: ON" << std::endl;
  else
    os << indent << "MultiThreadedExecution: OFF" << std::endl;

  os << indent << "NormalizeAcrossScale: " << m_NormalizeAcrossScale << std::endl;
}

} // end namespace itk

#endif
