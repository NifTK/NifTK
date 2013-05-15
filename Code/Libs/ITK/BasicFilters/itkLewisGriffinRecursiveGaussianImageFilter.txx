/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkLewisGriffinRecursiveGaussianImageFilter_txx
#define __itkLewisGriffinRecursiveGaussianImageFilter_txx

#include "itkLewisGriffinRecursiveGaussianImageFilter.h"
#include <itkObjectFactory.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageLinearConstIteratorWithIndex.h>
#include <itkProgressReporter.h>
#include <new>


namespace itk
{
  
template <typename TInputImage, typename TOutputImage>
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::LewisGriffinRecursiveGaussianImageFilter()
{
  this->SetNumberOfRequiredOutputs( 1 );
  this->SetNumberOfRequiredInputs( 1 );

  m_Direction = 0;
  m_Sigma = 1.0;
  m_Order = ZeroOrder;

  m_Kernel = 0;
  m_KernelSize = 0;
}
  
template <typename TInputImage, typename TOutputImage>
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::~LewisGriffinRecursiveGaussianImageFilter()
{
  if (m_Kernel)
    delete[] m_Kernel;
}


/**
 * Set Input Image
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::SetInputImage( const TInputImage * input )
{
  // ProcessObject is not const_correct so this const_cast is required
  ProcessObject::SetNthInput(0, 
                             const_cast< TInputImage * >(input) );
}


/**
 * Get Input Image
 */
template <typename TInputImage, typename TOutputImage>
const TInputImage *
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::GetInputImage( void )
{
  return dynamic_cast<const TInputImage *>(
    (ProcessObject::GetInput(0)));
}



/**
 *   Explicitly set a zeroth order derivative.
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::SetZeroOrder()
{
  this->SetOrder( ZeroOrder );
}

/**
 *   Explicitly set a first order derivative.
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::SetFirstOrder()
{
  this->SetOrder( FirstOrder );
}

/**
 *   Explicitly set a second order derivative.
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::SetSecondOrder()
{
  this->SetOrder( SecondOrder );
}

/**
 * Apply Recursive Filter
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::FilterDataArray(RealType *dst, RealType *src, int dx)
{
  // Assumes src and dst same size and kernel is odd sized.
  int j, k, os, dk_2 = (int) m_KernelSize/2;
  RealType tmp, *srcOrg=src, *dstOrg=dst;

  // Do non-boundary
  dst+=dk_2;

  for (j=0; j<(dx - ((int) m_KernelSize) + 1); j++) {
    tmp = 0;
    for (k=0; k<((int) m_KernelSize); k++)
      tmp += m_Kernel[k]* *(src + j+k);
    *(dst + j) = tmp;
  }


  // Do left boundary - cyclic
  src=srcOrg, dst=dstOrg;

  for (j=0; j<dk_2; j++) {
    tmp=0;
    for (k=-dk_2; k<dk_2+1; k++) {
      os=j+k;
      if (os<0)
	os +=dx;
      tmp += m_Kernel[k+dk_2]* *(src+os);
    }
    *(dst + j) = tmp;
  }


  // Do right boundary - cyclic
  src=srcOrg, dst=dstOrg+dk_2;

  for (j=dx - ((int) m_KernelSize) + 1; j<dx-dk_2; j++) {
    tmp=0;
    for (k=0; k<((int) m_KernelSize); k++) {
      os=j+k;
      if (os>=dx)
	os -=dx;
      tmp += m_Kernel[k]* *(src+os);
    }
    *(dst + j) = tmp;
  }

}


//
// we need all of the image in just the "Direction" we are separated into
//
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out)
    {
    OutputImageRegionType outputRegion = out->GetRequestedRegion();
    const OutputImageRegionType &largestOutputRegion = out->GetLargestPossibleRegion();

    // verify sane parameter
    if ( this->m_Direction >=  outputRegion.GetImageDimension() )
      {
      itkExceptionMacro("Direction selected for filtering is greater than ImageDimension")
      }

    // expand output region to match largest in the "Direction" dimension
    outputRegion.SetIndex( m_Direction, largestOutputRegion.GetIndex(m_Direction) );
    outputRegion.SetSize( m_Direction, largestOutputRegion.GetSize(m_Direction) );
    
    out->SetRequestedRegion( outputRegion );
    }
}


template <typename TInputImage, typename TOutputImage>
int
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion)
{
  // Get the output pointer
  OutputImageType * outputPtr = this->GetOutput();
  const typename TOutputImage::SizeType& requestedRegionSize 
    = outputPtr->GetRequestedRegion().GetSize();

  int splitAxis;
  typename TOutputImage::IndexType splitIndex;
  typename TOutputImage::SizeType splitSize;

  // Initialize the splitRegion to the output requested region
  splitRegion = outputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  // and avoid the current dimension
  splitAxis = outputPtr->GetImageDimension() - 1;
  while (requestedRegionSize[splitAxis] == 1 || splitAxis == (int)m_Direction)
    {
    --splitAxis;
    if (splitAxis < 0)
      { // cannot split
      itkDebugMacro("  Cannot Split");
      return 1;
      }
    }

  // determine the actual number of pieces that will be generated
  typename TOutputImage::SizeType::SizeValueType range = requestedRegionSize[splitAxis];
  int valuesPerThread = (int)vcl_ceil(range/(double)num);
  int maxThreadIdUsed = (int)vcl_ceil(range/(double)valuesPerThread) - 1;

  // Split the region
  if (i < maxThreadIdUsed)
    {
    splitIndex[splitAxis] += i*valuesPerThread;
    splitSize[splitAxis] = valuesPerThread;
    }
  if (i == maxThreadIdUsed)
    {
    splitIndex[splitAxis] += i*valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[splitAxis] = splitSize[splitAxis] - i*valuesPerThread;
    }
  
  // set the split region ivars
  splitRegion.SetIndex( splitIndex );
  splitRegion.SetSize( splitSize );

  itkDebugMacro("  Split Piece: " << splitRegion );

  return maxThreadIdUsed + 1;
}


template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData()
{
  typedef ImageRegion< TInputImage::ImageDimension > RegionType;
    
  typename TInputImage::ConstPointer   inputImage(    this->GetInputImage ()   );
  typename TOutputImage::Pointer       outputImage(   this->GetOutput()        );

  const unsigned int imageDimension = inputImage->GetImageDimension();

  if( this->m_Direction >= imageDimension )
    {
    itkExceptionMacro("Direction selected for filtering is greater than ImageDimension");
    }

  const typename InputImageType::SpacingType & pixelSize
    = inputImage->GetSpacing();
  
  this->SetUp( pixelSize[m_Direction] );
  
  RegionType region = outputImage->GetRequestedRegion();

  const unsigned int ln = region.GetSize()[ this->m_Direction ];

  if( ln < 4 )
    {
    itkExceptionMacro("The number of pixels along direction " << this->m_Direction << " is less than 4. This filter requires a minimum of four pixels along the dimension to be processed.");
    }

}


/**
 * Compute Recursive filter
 * line by line in one of the dimensions
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId)
{
  typedef typename TOutputImage::PixelType  OutputPixelType;

  typedef ImageLinearConstIteratorWithIndex< TInputImage >  InputConstIteratorType;
  typedef ImageLinearIteratorWithIndex< TOutputImage >      OutputIteratorType;

  typedef ImageRegion< TInputImage::ImageDimension > RegionType;
    
  typename TInputImage::ConstPointer   inputImage(    this->GetInputImage ()   );
  typename TOutputImage::Pointer       outputImage(   this->GetOutput()        );
    
  RegionType region = outputRegionForThread;

  InputConstIteratorType  inputIterator(  inputImage,  region );
  OutputIteratorType      outputIterator( outputImage, region );

  inputIterator.SetDirection(  this->m_Direction );
  outputIterator.SetDirection( this->m_Direction );

  
  const unsigned int ln = region.GetSize()[ this->m_Direction ];

  RealType *inps = 0;
  RealType *outs = 0;

  try 
    {
    inps = new RealType[ ln ];
    }
  catch( std::bad_alloc & ) 
    {
    itkExceptionMacro("Problem allocating memory for internal computations");
    }

  try 
    {
    outs = new RealType[ ln ];
    }
  catch( std::bad_alloc & ) 
    {
    delete [] inps;
    itkExceptionMacro("Problem allocating memory for internal computations");
    }

  inputIterator.GoToBegin();
  outputIterator.GoToBegin();

  const typename TInputImage::OffsetValueType * offsetTable = inputImage->GetOffsetTable();
  
  const unsigned int numberOfLinesToProcess = offsetTable[ TInputImage::ImageDimension ] / ln;
  ProgressReporter progress(this, threadId, numberOfLinesToProcess, 10 );


  try  // this try is intended to catch an eventual AbortException.
    {
    while( !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd() )
      {
      unsigned int i=0;
      while( !inputIterator.IsAtEndOfLine() )
        {
        inps[i++]      = inputIterator.Get();
        ++inputIterator;
        }

      this->FilterDataArray( outs, inps, (int) ln );

      unsigned int j=0; 
      while( !outputIterator.IsAtEndOfLine() )
        {
        outputIterator.Set( static_cast<OutputPixelType>( outs[j++] ) );
        ++outputIterator;
        }

      inputIterator.NextLine();
      outputIterator.NextLine();

      // Although the method name is CompletedPixel(),
      // this is being called after each line is processed
      progress.CompletedPixel();  
      }
    }
  catch( ProcessAborted  & )
    {
    // User aborted filter excecution Here we catch an exception thrown by the
    // progress reporter and rethrow it with the correct line number and file
    // name. We also invoke AbortEvent in case some observer was interested on
    // it.
    // release locally allocated memory
    delete [] outs;
    delete [] inps;
    // Throw the final exception.
    ProcessAborted e(__FILE__,__LINE__);
    e.SetDescription("Process aborted.");
    e.SetLocation(ITK_LOCATION);
    throw e;
    }

  delete [] outs;
  delete [] inps;
}



/**
 *   Compute Zero Order Gaussian.
 */
template <typename TInputImage, typename TOutputImage>
typename LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>::RealType
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::GaussianZeroOrder(RealType x, RealType sigma)
{ 
  return (exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * vnl_math::pi) * sigma));
}



/**
 *   Compute First Order Gaussian.
 */
template <typename TInputImage, typename TOutputImage>
typename LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>::RealType
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::GaussianFirstOrder(RealType x, RealType sigma)
{ 
  
  return (exp(-(x * x) / (2 * sigma * sigma)) * x / (sqrt(2 * vnl_math::pi)* sigma * sigma * sigma));
}



/**
 *   Compute Second Order Gaussian.
 */
template <typename TInputImage, typename TOutputImage>
typename LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>::RealType
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::GaussianSecondOrder(RealType x, RealType sigma)
{ 
  return (exp(-(x * x) / (2 * sigma * sigma)) * (x * x - sigma * sigma) / (sqrt(2 * vnl_math::pi) * sigma * sigma * sigma * sigma * sigma));
}



/**
 *   Compute filter for Gaussian kernel.
 */
template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::SetUp(ScalarRealType spacing)
{ 

  RealType sigma = m_Sigma/spacing;
  m_KernelSize = (unsigned int) (ceil(sigma)*5*2 + 1);
  RealType end = (RealType) ((m_KernelSize - 1) / 2);
  RealType x;

  if (m_Kernel) delete[] m_Kernel;
  m_Kernel = new RealType[m_KernelSize];

  switch( m_Order ) 
    {
    case ZeroOrder:
      {

	for (unsigned int i=0; i<m_KernelSize; i++) {
	  x = i - end;
	  m_Kernel[i] = GaussianZeroOrder(x, sigma);
	}

      break;
      }

    case FirstOrder:
      {
	for (unsigned int i=0; i<m_KernelSize; i++) {
	  x = i - end;
	  m_Kernel[i] = GaussianFirstOrder(x, sigma);
	}

	break;
      }

    case SecondOrder:
      {
	for (unsigned int i=0; i<m_KernelSize; i++) {
	  x = i - end;
	  m_Kernel[i] = GaussianSecondOrder(x, sigma);
	}

      break;
      }

    default:
      {
      itkExceptionMacro(<<"Unknown Order");
      return;
      }
    }


  // Normalise the kernel

  RealType sum = 0;

  for (unsigned int i=0; i<m_KernelSize; i++)
    sum += m_Kernel[i];

  for (unsigned int i=0; i<m_KernelSize; i++)
    if (m_Order == ZeroOrder)
      m_Kernel[i] /= sum;
    else
      m_Kernel[i] -= sum/m_KernelSize;

}


template <typename TInputImage, typename TOutputImage>
void
LewisGriffinRecursiveGaussianImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Direction: " << m_Direction << std::endl;
  os << "Sigma: " << m_Sigma << std::endl; 
  os << "Order: " << m_Order << std::endl; 
}

} // end namespace itk

#endif
