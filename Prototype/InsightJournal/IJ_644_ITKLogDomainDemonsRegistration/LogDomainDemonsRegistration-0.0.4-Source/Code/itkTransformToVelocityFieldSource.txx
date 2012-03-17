#ifndef __itkTransformToVelocityFieldSource_txx
#define __itkTransformToVelocityFieldSource_txx

#include "itkTransformToVelocityFieldSource.h"

#include "itkIdentityTransform.h"
#include "itkProgressReporter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkTranslationTransform.h"
#include "itkMatrixOffsetTransformBase.h"

#include "vnl_sd_matrix_tools.h"

namespace itk
{

// Constructor
template <class TOutputImage, class TTransformPrecisionType>
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::TransformToVelocityFieldSource()
{
  this->m_OutputSpacing.Fill(1.0);
  this->m_OutputOrigin.Fill(0.0);
  this->m_OutputDirection.SetIdentity();

  SizeType size;
  size.Fill( 0 );
  this->m_OutputRegion.SetSize( size );
  
  IndexType index;
  index.Fill( 0 );
  this->m_OutputRegion.SetIndex( index );
  
  this->m_Transform
    = IdentityTransform<TTransformPrecisionType, ImageDimension>::New();
  this->m_IncrementalTransform = 0;
}


// Print out a description of self
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  
  os << indent << "OutputRegion: " << this->m_OutputRegion << std::endl;
  os << indent << "OutputSpacing: " << this->m_OutputSpacing << std::endl;
  os << indent << "OutputOrigin: " << this->m_OutputOrigin << std::endl;
  os << indent << "OutputDirection: " << this->m_OutputDirection << std::endl;
  os << indent << "Transform: " << this->m_Transform.GetPointer() << std::endl;
  os << indent << "IncrementalTransform: "
     << this->m_IncrementalTransform.GetPointer() << std::endl;
}


// Set the output image size.
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::SetOutputSize( const SizeType & size )
{
  this->m_OutputRegion.SetSize( size );
}


// Get the output image size.
template <class TOutputImage, class TTransformPrecisionType>
const typename TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::SizeType &
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::GetOutputSize()
{
  return this->m_OutputRegion.GetSize();
}


// Set the output image index.
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::SetOutputIndex( const IndexType & index )
{
  this->m_OutputRegion.SetIndex( index );
}


// Get the output image index.
template <class TOutputImage, class TTransformPrecisionType>
const typename TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::IndexType &
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::GetOutputIndex()
{
  return this->m_OutputRegion.GetIndex();
}


// Set the output image spacing.
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::SetOutputSpacing( const double* spacing )
{
  SpacingType s( spacing );
  this->SetOutputSpacing( s );
}


// Set the output image origin.
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::SetOutputOrigin( const double* origin )
{
  OriginType p( origin );
  this->SetOutputOrigin( p );
}

// Helper method to set the output parameters based on this image
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::SetOutputParametersFromImage ( const ImageBaseType * image )
{
  if( !image )
    {
    itkExceptionMacro(<< "Cannot use a null image reference");
    }
  
  this->SetOutputOrigin( image->GetOrigin() );
  this->SetOutputSpacing( image->GetSpacing() );
  this->SetOutputDirection( image->GetDirection() );
  this->SetOutputRegion( image->GetLargestPossibleRegion() );
}


// Set up state of filter before multi-threading.
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::BeforeThreadedGenerateData( void )
{
  this->m_IncrementalTransform = 0;
  
  if( !this->m_Transform )
    {
    itkExceptionMacro(<< "Transform not set");
    }

  typedef IdentityTransform<TTransformPrecisionType, ImageDimension> IdTrsfType;
  typedef TranslationTransform<TTransformPrecisionType, ImageDimension> TransTrsfType;
  typedef MatrixOffsetTransformBase<TTransformPrecisionType, ImageDimension, ImageDimension> MatOffTrsfType;

  const IdTrsfType * idtrsf = dynamic_cast<const IdTrsfType*>( this->m_Transform.GetPointer() );
  const TransTrsfType * transtrsf = dynamic_cast<const TransTrsfType*>( this->m_Transform.GetPointer() );
  if( idtrsf || transtrsf )
    {
    // Translation transform logarithms have a zero matrix
    // but keep the initial translation
    typename MatOffTrsfType::MatrixType matrixlog;
    typename MatOffTrsfType::OutputVectorType offsetlog;
    matrixlog.Fill(0.0);

    if( transtrsf )
      {
      offsetlog = transtrsf->GetOffset();
      }
    else
      {
      offsetlog.Fill(0.0);
      }
       
    typename MatOffTrsfType::Pointer trsf = MatOffTrsfType::New();
    trsf->SetMatrix(matrixlog);
    trsf->SetOffset(offsetlog);

    this->m_IncrementalTransform = trsf;

    itkDebugMacro(<< "Transform: " << this->m_Transform);
    itkDebugMacro(<< "IncrementalTransform: " << this->m_IncrementalTransform);
    
    return;
    }

  const MatOffTrsfType * mattrsf = dynamic_cast<const MatOffTrsfType*>( this->m_Transform.GetPointer() );
  if( mattrsf )
    {
    const typename MatOffTrsfType::MatrixType & matrix = mattrsf->GetMatrix();
    const typename MatOffTrsfType::OutputVectorType & offset = mattrsf->GetOffset();

    const unsigned int nr = MatOffTrsfType::MatrixType::RowDimensions;
    const unsigned int nc = MatOffTrsfType::MatrixType::ColumnDimensions;

    // Copy information into the homogeneous matrix
    vnl_matrix<double> homogmat(nr+1, nc+1, 0.0);
    for (unsigned int r = 0; r<nr; ++r)
      {
      for (unsigned int c = 0; c<nc; ++c)
        {
           homogmat(r,c) = matrix(r,c);
        }

      homogmat(r,nc) = offset[r];
      }
    homogmat(nr,nc) = 1.0;

    // Compute homogeneous matrix logarithm
    const vnl_matrix<double> loghomogmat = sdtools::GetLogarithm( homogmat );

    // Copy information back from the matrix logarithm
    typename MatOffTrsfType::MatrixType matrixlog;
    typename MatOffTrsfType::OutputVectorType offsetlog;
    for (unsigned int r = 0; r<nr; ++r)
      {
      for (unsigned int c = 0; c<nc; ++c)
        {
           matrixlog(r,c) = loghomogmat(r,c);
        }

      offsetlog[r] = loghomogmat(r,nc);
      }
    
    typename MatOffTrsfType::Pointer trsf = MatOffTrsfType::New();
    // Calling trsf->SetCenter( mattrsf->GetCenter() ); should be useless
    trsf->SetMatrix(matrixlog);
    trsf->SetOffset(offsetlog);
    
    this->m_IncrementalTransform = trsf;

    itkDebugMacro(<< "Transform: " << this->m_Transform);
    itkDebugMacro(<< "IncrementalTransform: " << this->m_IncrementalTransform);
    
    return;
    }

  
  if ( this->m_Transform->IsLinear() )
    {
    itkExceptionMacro(<< "The transform reports to be linear but is not a subclass of MatrixOffsetTransformBase, TranslationTransform or IdentityTransform. This case is not supported yet.");
    }

  itkExceptionMacro(<< "Only linear transforms are supported.");
}


// ThreadedGenerateData
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::ThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  int threadId )
{
  // Check whether we can use a fast path for resampling. Fast path
  // can be used if the transformation is linear. Transform respond
  // to the IsLinear() call.
  if ( this->m_IncrementalTransform->IsLinear() )
    {
    this->LinearThreadedGenerateData( outputRegionForThread, threadId );
    return;
    }

  itkExceptionMacro(<< "Only linear transforms are supported.");
}


template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::LinearThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  int threadId )
{
  // Get the output pointer
  OutputImagePointer      outputPtr = this->GetOutput();

  // Create an iterator that will walk the output region for this thread.
  typedef ImageLinearIteratorWithIndex<TOutputImage> OutputIteratorType;
  OutputIteratorType outIt( outputPtr, outputRegionForThread );
  outIt.SetDirection( 0 );

  // Define a few indices that will be used to translate from an input pixel
  // to an output pixel
  PointType outputPoint;         // Coordinates of current output pixel
  PointType transformedPoint;    // Coordinates of transformed pixel
  
  IndexType index;

  // Support for progress methods/callbacks
  ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );
    
  // Determine the position of the first pixel in the scanline
  outIt.GoToBegin();
  index = outIt.GetIndex();
  outputPtr->TransformIndexToPhysicalPoint( index, outputPoint );
  
  // Compute corresponding transformed pixel position
  transformedPoint = this->m_IncrementalTransform->TransformPoint( outputPoint );
  
  // Compare with the ResampleImageFilter

  // Compute delta
  PointType outputPointNeighbour;
  PointType transformedPointNeighbour;
  typedef typename PointType::VectorType VectorType;
  VectorType delta;
  ++index[0];
  outputPtr->TransformIndexToPhysicalPoint( index, outputPointNeighbour );
  transformedPointNeighbour = this->m_IncrementalTransform->TransformPoint( outputPointNeighbour );
  delta = transformedPointNeighbour - transformedPoint;

  // loop over the vector image
  while ( !outIt.IsAtEnd() )
    {
    // Get current point
    index = outIt.GetIndex();
    outputPtr->TransformIndexToPhysicalPoint( index, outputPoint );

    // Compute transformed point
    transformedPoint = this->m_IncrementalTransform->TransformPoint( outputPoint );

    while ( !outIt.IsAtEndOfLine() )
      {
      // Compute the deformation
      for ( unsigned int i = 0; i < ImageDimension; ++i )
        {
        outIt.Value()[i] = static_cast<PixelValueType>( transformedPoint[i] );
        }

      // Update stuff
      progress.CompletedPixel();
      ++outIt;
      transformedPoint += delta;
      }

    outIt.NextLine();
    }
}


// Inform pipeline of required output region
template <class TOutputImage, class TTransformPrecisionType>
void 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::GenerateOutputInformation( void )
{
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointer to the output
  OutputImagePointer outputPtr = this->GetOutput();
  if ( !outputPtr )
    {
    return;
    }

  outputPtr->SetLargestPossibleRegion( m_OutputRegion );

  outputPtr->SetSpacing( m_OutputSpacing );
  outputPtr->SetOrigin( m_OutputOrigin );
  outputPtr->SetDirection( m_OutputDirection );
}


// Verify if any of the components has been modified.
template <class TOutputImage, class TTransformPrecisionType>
unsigned long 
TransformToVelocityFieldSource<TOutputImage,TTransformPrecisionType>
::GetMTime( void ) const
{
  unsigned long latestTime = Object::GetMTime(); 

  if( this->m_Transform )
    {
    if( latestTime < this->m_Transform->GetMTime() )
      {
      latestTime = this->m_Transform->GetMTime();
      }
    }

  return latestTime;
}


} // end namespace itk

#endif // end #ifndef _itkTransformToVelocityFieldSource_txx
