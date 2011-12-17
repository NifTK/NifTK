/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkBasicImageFeaturesImageFilter_txx
#define __itkBasicImageFeaturesImageFilter_txx

#include "itkBasicImageFeaturesImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkImageDuplicator.h"
#include "itkProgressReporter.h"
#include "ConversionUtils.h"

#include <vnl/vnl_double_2x2.h>

#include "itkUCLMacro.h"


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::BasicImageFeaturesImageFilter()
{

  // Multi-threaded execution is enabled by default
  m_FlagMultiThreadedExecution = true;

  m_Epsilon = 1.0e-05;
  m_NormalizeAcrossScale = false;
  m_FlagCalculateOrientatedBIFs = false;
  m_FlagLocalOrientationSet = false;
  m_FlagLinesOnly = false;

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
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::~BasicImageFeaturesImageFilter()
{
}

/* -----------------------------------------------------------------------
   SetSigma()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
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
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
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
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out)
    {
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
    }
}


/* -----------------------------------------------------------------------
   GetDerivative()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
typename BasicImageFeaturesImageFilter<TInputImage,TOutputImage>::RealImagePointer 
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::GetDerivative( DerivativeFilterOrderEnumTypeX xOrder,
		 DerivativeFilterOrderEnumTypeY yOrder )
{
  DerivativeFilterPointerX derivativeFilterX;
  DerivativeFilterPointerY derivativeFilterY;

  derivativeFilterX = DerivativeFilterTypeX::New();
  derivativeFilterY = DerivativeFilterTypeY::New();

  derivativeFilterX->SetSigma( m_Sigma );
  derivativeFilterY->SetSigma( m_Sigma );

  derivativeFilterX->SetInput( this->GetInput() );
  derivativeFilterY->SetInput( derivativeFilterX->GetOutput() );

  derivativeFilterX->SetDirection( 0 );
  derivativeFilterY->SetDirection( 1 );

  derivativeFilterX->SetOrder( xOrder );
  derivativeFilterY->SetOrder( yOrder );

  derivativeFilterY->Update(); 

  return derivativeFilterY->GetOutput();
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData(void)
{
  typedef itk::ImageFileWriter< RealImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();

  // S00

  niftkitkInfoMacro(<< "Computing S00");

  m_S00 = GetDerivative(DerivativeFilterTypeX::ZeroOrder,
			DerivativeFilterTypeY::ZeroOrder );

  // S10

  niftkitkInfoMacro(<< "Computing S10");

  m_S10 = GetDerivative(DerivativeFilterTypeX::FirstOrder,
			DerivativeFilterTypeY::ZeroOrder );

  // S01

  niftkitkInfoMacro(<< "Computing S01");

  m_S01 = GetDerivative(DerivativeFilterTypeX::ZeroOrder,
			DerivativeFilterTypeY::FirstOrder );

  // S11

  niftkitkInfoMacro(<< "Computing S11");

  m_S11 = GetDerivative(DerivativeFilterTypeX::FirstOrder,
			DerivativeFilterTypeY::FirstOrder );

  // S20

  niftkitkInfoMacro(<< "Computing S02");

  m_S20 = GetDerivative(DerivativeFilterTypeX::SecondOrder,
			DerivativeFilterTypeY::ZeroOrder );

  // S02

  niftkitkInfoMacro(<< "Computing S20");

  m_S02 = GetDerivative(DerivativeFilterTypeX::ZeroOrder,
			DerivativeFilterTypeY::SecondOrder );


  // Allocate the filter response images

  m_ResponseFlat      = RealImageType::New();
  m_ResponseSlope     = RealImageType::New();
  m_ResponseDarkBlob  = RealImageType::New();
  m_ResponseLightBlob = RealImageType::New();
  m_ResponseDarkLine  = RealImageType::New();
  m_ResponseLightLine = RealImageType::New();
  m_ResponseSaddle    = RealImageType::New();
  m_FirstOrderOrientation     = RealImageType::New();
  m_SecondOrderOrientation    = RealImageType::New();

  typename RealImageType::RegionType region = this->GetInput()->GetLargestPossibleRegion();

  m_ResponseFlat->SetRegions( region );
  m_ResponseSlope->SetRegions( region );
  m_ResponseDarkBlob->SetRegions( region );
  m_ResponseLightBlob->SetRegions( region );
  m_ResponseDarkLine->SetRegions( region );
  m_ResponseLightLine->SetRegions( region );
  m_ResponseSaddle->SetRegions( region );
  m_FirstOrderOrientation->SetRegions( region );
  m_SecondOrderOrientation->SetRegions( region );

  m_ResponseFlat->SetSpacing( this->GetInput()->GetSpacing() );
  m_ResponseSlope->SetSpacing( this->GetInput()->GetSpacing() );
  m_ResponseDarkBlob->SetSpacing( this->GetInput()->GetSpacing() );
  m_ResponseLightBlob->SetSpacing( this->GetInput()->GetSpacing() );
  m_ResponseDarkLine->SetSpacing( this->GetInput()->GetSpacing() );
  m_ResponseLightLine->SetSpacing( this->GetInput()->GetSpacing() );
  m_ResponseSaddle->SetSpacing( this->GetInput()->GetSpacing() );
  m_FirstOrderOrientation->SetSpacing( this->GetInput()->GetSpacing() );
  m_SecondOrderOrientation->SetSpacing( this->GetInput()->GetSpacing() );

  m_ResponseFlat->SetOrigin( this->GetInput()->GetOrigin() );
  m_ResponseSlope->SetOrigin( this->GetInput()->GetOrigin() );
  m_ResponseDarkBlob->SetOrigin( this->GetInput()->GetOrigin() );
  m_ResponseLightBlob->SetOrigin( this->GetInput()->GetOrigin() );
  m_ResponseDarkLine->SetOrigin( this->GetInput()->GetOrigin() );
  m_ResponseLightLine->SetOrigin( this->GetInput()->GetOrigin() );
  m_ResponseSaddle->SetOrigin( this->GetInput()->GetOrigin() );
  m_FirstOrderOrientation->SetOrigin( this->GetInput()->GetOrigin() );
  m_SecondOrderOrientation->SetOrigin( this->GetInput()->GetOrigin() );

  m_ResponseFlat->Allocate( );
  m_ResponseSlope->Allocate( );
  m_ResponseDarkBlob->Allocate( );
  m_ResponseLightBlob->Allocate( );
  m_ResponseDarkLine->Allocate( );
  m_ResponseLightLine->Allocate( );
  m_ResponseSaddle->Allocate( );
  m_FirstOrderOrientation->Allocate( );
  m_SecondOrderOrientation->Allocate( );

  m_ResponseFlat->FillBuffer( 0. );
  m_ResponseSlope->FillBuffer( 0. );
  m_ResponseDarkBlob->FillBuffer( 0. );
  m_ResponseLightBlob->FillBuffer( 0. );
  m_ResponseDarkLine->FillBuffer( 0. );
  m_ResponseLightLine->FillBuffer( 0. );
  m_ResponseSaddle->FillBuffer( 0. );
  m_FirstOrderOrientation->FillBuffer( 0. );
  m_SecondOrderOrientation->FillBuffer( 0. );
}




/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
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
    typename ImageSource<TOutputImage>::ThreadStruct str;
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
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       int threadId)
{
  unsigned int *adjust = 0;
  unsigned int *reorder = 0;
  vnl_double_2 *dirn = 0;		///< The list of vector orientations
  vnl_double_2 *orient = 0;		///< The list of vector orientations for the current pixel (eg. wrt. to origin)


  // Array to ensure oriented BIFs are corectly ordered

  if (! adjust) {
    adjust = new unsigned int[7];

    adjust[0] = 0;                // flat = 0 
    adjust[1] = 0;                // slope = 1 .. 1+n-1
    adjust[2] = m_NumberOfOrientations - 1; // min = 2+n-1
    adjust[3] = m_NumberOfOrientations - 1; // max = 3+n-1
    adjust[4] = m_NumberOfOrientations - 1; // light line = 4+n-1 ..
    adjust[5] = 3*m_NumberOfOrientations/2 - 2; // dark line = 5+n-1+n/2-1 ..
    adjust[6] = 2*m_NumberOfOrientations - 3; // saddle = 6+n-1+n/2-1+n/2-1 .. 
  }

  // Array to rotate the light line indices so they're correctly
  // aligned with the dark line indices

  if (! reorder) {
    reorder = new unsigned int[m_NumberOfOrientations/2];

    unsigned int i, j;
    for (i=0, j=m_NumberOfOrientations/4; i<m_NumberOfOrientations/2; i++) {
      reorder[i] = j;
      j++;
      if (j > m_NumberOfOrientations/2 - 1)
        j = 0;
    }
  }

  if ( ! dirn )
    dirn = new vnl_double_2[m_NumberOfOrientations];

  if ( ! orient )
    orient = new vnl_double_2[m_NumberOfOrientations];

  unsigned int k;
  double theta;
  vnl_double_2x2 R;

  dirn[0][0] = 1.; dirn[0][1] = 0.;

  for ( k=1; k<m_NumberOfOrientations; k++ ) {

    theta = k*2.*M_PI/m_NumberOfOrientations;

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

  ImageRegionIterator< RealImageType > itS00
    = ImageRegionIterator< RealImageType >( m_S00, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS10
    = ImageRegionIterator< RealImageType >( m_S10, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS01
    = ImageRegionIterator< RealImageType >( m_S01, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS11
    = ImageRegionIterator< RealImageType >( m_S11, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS20
    = ImageRegionIterator< RealImageType >( m_S20, outputRegionForThread );
  ImageRegionIterator< RealImageType > itS02
    = ImageRegionIterator< RealImageType >( m_S02, outputRegionForThread );


  ImageRegionIterator< RealImageType > itFlat
    = ImageRegionIterator< RealImageType >( m_ResponseFlat, outputRegionForThread );
  ImageRegionIterator< RealImageType > itSlope
    = ImageRegionIterator< RealImageType >( m_ResponseSlope, outputRegionForThread );
  ImageRegionIterator< RealImageType > itDarkBlob
    = ImageRegionIterator< RealImageType >( m_ResponseDarkBlob, outputRegionForThread );
  ImageRegionIterator< RealImageType > itLightBlob
    = ImageRegionIterator< RealImageType >( m_ResponseLightBlob, outputRegionForThread );
  ImageRegionIterator< RealImageType > itDarkLine
    = ImageRegionIterator< RealImageType >( m_ResponseDarkLine, outputRegionForThread );
  ImageRegionIterator< RealImageType > itLightLine
    = ImageRegionIterator< RealImageType >( m_ResponseLightLine, outputRegionForThread );
  ImageRegionIterator< RealImageType > itSaddle
    = ImageRegionIterator< RealImageType >( m_ResponseSaddle, outputRegionForThread );

  ImageRegionIterator< RealImageType > itFirstOrderOrientation
    = ImageRegionIterator< RealImageType >( m_FirstOrderOrientation, outputRegionForThread );
  ImageRegionIterator< RealImageType > itSecondOrderOrientation
    = ImageRegionIterator< RealImageType >( m_SecondOrderOrientation, outputRegionForThread );


  outputIterator.GoToBegin();

  itS00.GoToBegin();
  itS10.GoToBegin();
  itS01.GoToBegin();
  itS11.GoToBegin();
  itS20.GoToBegin();
  itS02.GoToBegin();

  itFlat.GoToBegin();
  itSlope.GoToBegin();
  itDarkBlob.GoToBegin();
  itLightBlob.GoToBegin();
  itDarkLine.GoToBegin();
  itLightLine.GoToBegin();
  itSaddle.GoToBegin();

  itFirstOrderOrientation.GoToBegin();
  itSecondOrderOrientation.GoToBegin();

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
  RealType opts[7];

  for ( ; ! outputIterator.IsAtEnd(); ++outputIterator) {


    lambda = sigmaeSqr*(itS20.Get() + itS02.Get())/2.;
    gamma = sigmaeSqr*vcl_sqrt( ( itS20.Get() - itS02.Get() )*( itS20.Get() - itS02.Get() )/4. + itS11.Get()*itS11.Get() );

    opts[0] = m_Epsilon * itS00.Get();
    itFlat.Set( opts[0] );

    opts[1] = sigmae * vcl_sqrt( itS10.Get()*itS10.Get() + itS01.Get()*itS01.Get() );
    itSlope.Set( opts[1] );

    opts[2] = lambda;
    itDarkBlob.Set( opts[2] );

    opts[3] = -lambda;
    itLightBlob.Set( opts[3] );

    opts[4] = oneSqrtTwo*( gamma - lambda );
    itDarkLine.Set( opts[4] );

    opts[5] = oneSqrtTwo*( gamma + lambda );
    itLightLine.Set( opts[5] );

    opts[6] = gamma;
    itSaddle.Set( opts[6] );

    maximum = opts[0]; 
    idx = 0;

#if 0
    index = outputIterator.GetIndex();

    if ( (index[0] == 106) && (index[1]== 128)) 
	std::cout << "[" << index[0] << ", " << index[1] << "]"
		  << " S00=" << itS00.Get() 
		  << " S10=" << itS10.Get() 
		  << " S01=" << itS01.Get() 
		  << " S11=" << itS11.Get() 
		  << " S20=" << itS20.Get() 
		  << " S02=" << itS02.Get() 
		  << " sigma=" << sigma
		  << " sigmae=" << sigmae
		  << " opts[0]=" << opts[0]
		  << " opts[1]=" << opts[1]
		  << " opts[2]=" << opts[2]
		  << " opts[3]=" << opts[3]
		  << " opts[4]=" << opts[4]
		  << " opts[5]=" << opts[5]
		  << " opts[6]=" << opts[6]
		  << std::endl;
#endif

    for (k=1; k<7; k++) {
      tmp = opts[k];
      
      if (tmp >= maximum) {
	maximum  = tmp;
	idx = k;
      }
    }

    if ( m_FlagLinesOnly && (idx != 4) && (idx != 5) )
      idx = 0;


    // Calculate orientated BIFs?
    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~

    My preferred method of calculating a quantized 1st order direction
    is to form g, the vector of 1st derivatives, and see which is the
    largest of g.v when v can be

    {1,0}, {root2,root2}, {0,1}, {-root2,root2}, {-1,0},
    {-root2,-root2}, {0,-1}, {root2,-root2}

    For second order, I form the matrix S={{Ixx,Ixy},{Ixy,Iyy}} and
    compute v.S.v for the following v {1,0}, {root2,root2}, {0,1},
    {-root2,root2} and see which is the largest

    I would suggest the following modification. If the nipple is at
    {nx,ny} and the coordinates of the current pixel are {x,y}, make
    the 2D rotation matrix R that rotates {nx,ny}-{x,y} so that y=0,
    then apply R to each of the v before you then do the
    see-which-is-lrgest step as before.
    
    This is all a little bit slower than doing arctan and dealing
    angles etc. but it is bulletproof.

    Lewis Griffin */

    if (m_FlagCalculateOrientatedBIFs) {

      if ( m_FlagLocalOrientationSet || m_FlagOriginSet ) {

	OutputImagePointType point;

	// Calculate the orientation using a local reference orientation
      
	if (m_FlagLocalOrientationSet) {

	  index = outputIterator.GetIndex();
	  outImage->TransformIndexToPhysicalPoint( index, point );

	  typename RealImageType::IndexType orientIndex;
	  typename RealImageType::PointType orientPoint;

	  orientPoint = point;
	
	  if ( m_OrientationInX->TransformPhysicalPointToIndex( orientPoint, orientIndex ) ) {
		 
	    point[0] = m_OrientationInX->GetPixel( orientIndex );
	    point[1] = m_OrientationInY->GetPixel( orientIndex );
	  }
	}

	// Compute the orientation with respect to the origin

	else if (m_FlagOriginSet) {

	  index = outputIterator.GetIndex();

	  outImage->TransformIndexToPhysicalPoint( index, point );
	  point[0] -= m_Origin[0];
	  point[1] -= m_Origin[1];
	}


	double r = (point[0]*point[0] + point[1]*point[1]);
	
	if ( r != 0. ) {
	  r = vcl_sqrt( r );
	  point[0] /= r;
	  point[1] /= r;
	  
	  R( 0, 0 ) = point[0]; R( 0, 1 ) =  point[1]; 
	  R( 1, 0 ) = point[1]; R( 1, 1 ) = -point[0];
	}
	else {
	  R.set_identity();
	}

	for (k=0; k<m_NumberOfOrientations; k++) {
	  orient[k] = R*dirn[k];
	}
      }

      // Quantise the first order orientation

      vnl_double_2 g( itS10.Get(), itS01.Get() );
      g.normalize();

      double max, dotproduct;
      unsigned int cat1st;

      max = dot_product( g, orient[0] );
      cat1st = 0;

      for (k=1; k<m_NumberOfOrientations; k++) {
	dotproduct = dot_product( g, orient[k] );

	if (dotproduct > max) {
	  max = dotproduct;
	  cat1st = k;
	}
      }

      // Quantise the second order orientation

      vnl_double_2x2 S; 
      S( 0, 0 ) = itS20.Get(); S( 0, 1 ) = itS11.Get(); 
      S( 1, 0 ) = itS11.Get(); S( 1, 1 ) = itS02.Get();

      unsigned int cat2nd;

      max = dot_product( orient[0], S * orient[0] );
      cat2nd = 0;

      for (k=1; k<m_NumberOfOrientations/2; k++) {
	dotproduct = dot_product( orient[k], S * orient[k] );

	if (dotproduct > max) {
	  max = dotproduct;
	  cat2nd = k;
	}
      }


      // Adjust the orientations to the appropriate range according to
      // whether this is first or second order structure

      unsigned char tmp = adjust[idx];

      if (idx == 1)             // slope
	idx += cat1st;
      else
	if (idx == 4 || idx == 6) // dark line or saddle
	  idx += cat2nd;
	else
	  if (idx == 5)         // light line
	    idx += reorder[cat2nd];

      idx += tmp;
    }

    outputIterator.Set( idx );

    ++itS00;
    ++itS10;
    ++itS01;
    ++itS11;
    ++itS20;
    ++itS02;

    ++itFlat;
    ++itSlope;
    ++itDarkBlob;
    ++itLightBlob;
    ++itDarkLine;
    ++itLightLine;
    ++itSaddle;

    ++itFirstOrderOrientation;
    ++itSecondOrderOrientation;
  }

  if (adjust)  delete[] adjust;
  if (reorder) delete[] reorder;
  if (dirn)    delete[] dirn;
  if (orient)  delete[] orient;
}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
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
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
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

    case 0: { inputImage = m_S00; break; }
    case 1: { inputImage = m_S10; break; }
    case 2: { inputImage = m_S01; break; }
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

  std::cout << "Writing: " << filename;
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
   WriteFilterResponseToFile()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
::WriteFilterResponseToFile( int n, std::string filename ) 
{
  typename RealImageType::Pointer inputImage;

  typedef itk::ImageFileWriter< RealImageType > ResponseWriterType;
  typename ResponseWriterType::Pointer responseWriter = ResponseWriterType::New();

  typedef typename itk::MaskImageFilter< RealImageType, 
                                         MaskImageType, 
                                         RealImageType > ResponseMaskFilterType;
  typename ResponseMaskFilterType::Pointer maskFilter;


  switch ( n ) 
    {

    case 0: { inputImage = m_ResponseFlat;      break; }
    case 1: { inputImage = m_ResponseSlope;     break; }
    case 2: { inputImage = m_ResponseDarkBlob;  break; }
    case 3: { inputImage = m_ResponseLightBlob; break; }
    case 4: { inputImage = m_ResponseDarkLine;  break; }
    case 5: { inputImage = m_ResponseLightLine; break; }
    case 6: { inputImage = m_ResponseSaddle;    break; }
      
    default : {
      niftkitkErrorMacro(<< "Filter response number 'n' must satisfy: 0 < n < 7");
      return;
    }
    }

  if ( m_Mask ) {
    
    maskFilter = ResponseMaskFilterType::New();

    maskFilter->SetInput1( inputImage );
    maskFilter->SetInput2( m_Mask );

    responseWriter->SetInput( maskFilter->GetOutput() );
  }

  else 
    responseWriter->SetInput( inputImage );

  std::cout << "Writing: " << filename;
  responseWriter->SetFileName( filename.c_str() );

  try
    {
      responseWriter->Update();
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
BasicImageFeaturesImageFilter<TInputImage,TOutputImage>
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
