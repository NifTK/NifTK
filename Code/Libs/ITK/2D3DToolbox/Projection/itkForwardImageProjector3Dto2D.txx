/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkForwardImageProjector3Dto2D_txx
#define __itkForwardImageProjector3Dto2D_txx

#include "itkForwardImageProjector3Dto2D.h"

#include "itkImageRegionIterator.h"
#include "itkProgressReporter.h"

#include "itkCastImageFilter.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ForwardImageProjector3Dto2D<IntensityType>
::ForwardImageProjector3Dto2D()
{
  // Initialise the threshold above which intensities are integrated
  
  m_Threshold = 0.;

  // Multi-threaded execution is enabled by default

  m_FlagMultiThreadedExecution = true;

  // Set default values for the output image size

  m_OutputImageSize[0]  = 100;  // size along X
  m_OutputImageSize[1]  = 100;  // size along Y
 
  // Set default values for the output image resolution

  m_OutputImageSpacing[0]  = 1;  // resolution along X axis
  m_OutputImageSpacing[1]  = 1;  // resolution along Y axis
 
  // Set default values for the output image origin

  m_OutputImageOrigin[0]  = 0.;  // origin in X
  m_OutputImageOrigin[1]  = 0.;  // origin in Y

  OutputImagePointer      outputPtr = this->GetOutput();

  outputPtr->SetSpacing(m_OutputImageSpacing);
  outputPtr->SetOrigin(m_OutputImageOrigin);
}


/* -----------------------------------------------------------------------
   PrintSelf(std::ostream&, Indent)
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>::
PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Output image size: " << m_OutputImageSize << std::endl;
  os << indent << "Output image spacing: " << m_OutputImageSpacing << std::endl;
  os << indent << "Output image origin: " << m_OutputImageOrigin << std::endl;

  os << indent << "Threshold: " << m_Threshold << std::endl;

  if (m_FlagMultiThreadedExecution)
    os << indent << "MultiThreadedExecution: ON" << std::endl;
  else
    os << indent << "MultiThreadedExecution: OFF" << std::endl;
}


/* -----------------------------------------------------------------------
   GenerateOutputInformation()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::GenerateOutputInformation()
{
  OutputImageRegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize( m_OutputImageSize );

  OutputImagePointer outputPtr = this->GetOutput();
  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );  

  niftkitkDebugMacro(<< "Forward-projection output size: " << outputPtr->GetLargestPossibleRegion().GetSize());
}

/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::GenerateInputRequestedRegion()
{
  // generate everything in the region of interest
  InputImagePointer  inputPtr = const_cast<InputImageType *> (this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  niftkitkDebugMacro(<< "Forward-projection input size: " << inputPtr->GetLargestPossibleRegion().GetSize());
}

/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion(DataObject *)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass' implementation of this method
  Superclass::EnlargeOutputRequestedRegion(output);
  
  // generate everything in the region of interest
  this->GetOutput()->SetRequestedRegionToLargestPossibleRegion();
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::BeforeThreadedGenerateData(void)
{
#if 0
  static int n = 0;
  char filename[256];

  // First cast the image from double to float

  typedef itk::Image< float, 3 > OutputImageType;
  typedef itk::CastImageFilter< InputImageType, OutputImageType > CastFilterType;

  typename CastFilterType::Pointer  caster =  CastFilterType::New();

  caster->SetInput( this->GetInput() );

  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  sprintf(filename, "/tmp/ForwardImageProjector3Dto2D_INPUT_%03d.gipl", ++n );
  writer->SetFileName( filename );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << "BeforeThreadedGenerateData: " << filename << std::endl;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << filename << "; " << err << endl;
  }

  if (n >= 21)
    exit(1);
#endif
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::GenerateData(void)
{
  // Perform multi-threaded execution by default

  if (m_FlagMultiThreadedExecution) {
    
	niftkitkDebugMacro(<< "Multi-threaded forward-projection");

    Superclass::GenerateData();
  }

  // Single-threaded execution

  else {
  
	niftkitkDebugMacro(<< "Single-threaded forward-projection");

    // Call a method that can be overridden by a subclass to perform
    // some calculations prior to splitting the main computations into
    // separate threads
    this->BeforeThreadedGenerateData();
  
    double integral = 0;
    OutputImageIndexType outIndex;
    OutputImagePointType outPoint;

    ImageRegionIterator<OutputImageType> outputIterator;
  
    Ray<InputImageType> ray;
    Matrix<double, 4, 4> projMatrix;

 
    // Call a method that can be overriden by a subclass to allocate
    // memory for the filter's outputs
    this->AllocateOutputs();

    InputImageConstPointer inImage  = this->GetInput();
    OutputImagePointer     outImage = this->GetOutput();


    // Create the ray object

    ray.SetImage( inImage );

    projMatrix = this->m_PerspectiveTransform->GetMatrix();
    projMatrix *= this->m_AffineTransform->GetFullAffineMatrix();

    ray.SetProjectionMatrix(projMatrix);

    // Iterate over pixels in the 2D projection (i.e. output) image

    outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetRequestedRegion());

    for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) 
      {

	// Determine the coordinate of the output pixel
	outIndex = outputIterator.GetIndex();
	outImage->TransformIndexToPhysicalPoint(outIndex, outPoint);
      
	// Create a ray for this coordinate
	ray.SetRay(outPoint);
	ray.IntegrateAboveThreshold(integral, m_Threshold);

	outputIterator.Set( static_cast<IntensityType>( integral ) );
      }

    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    this->AfterThreadedGenerateData();
  }
}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const OutputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       int threadId)
{
  double integral = 0;
  OutputImageIndexType outIndex;
  OutputImagePointType outPoint;

  ImageRegionIterator<OutputImageType> outputIterator;
  
  Ray<InputImageType> ray;
  Matrix<double, 4, 4> projMatrix;

 
  // Allocate output

  InputImageConstPointer inImage  = this->GetInput();
  OutputImagePointer     outImage = this->GetOutput();

  // Support progress methods/callbacks

  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  // Create the ray object

  ray.SetImage( inImage );

  OutputImageSpacingType res2D = outImage->GetSpacing();

  ray.SetProjectionResolution2Dmm( res2D[0], res2D[1] );

  projMatrix = this->m_PerspectiveTransform->GetMatrix();
  projMatrix *= this->m_AffineTransform->GetFullAffineMatrix();

  ray.SetProjectionMatrix(projMatrix);

  // Iterate over pixels in the 2D projection (i.e. output) image

  outputIterator = ImageRegionIterator<OutputImageType>(outImage, outputRegionForThread);

  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) {

    // Determine the coordinate of the output pixel
    outIndex = outputIterator.GetIndex();
    outImage->TransformIndexToPhysicalPoint(outIndex, outPoint);
    
    // Create a ray for this coordinate
    ray.SetRay(outPoint);
    ray.IntegrateAboveThreshold(integral, m_Threshold);

    outputIterator.Set( static_cast<IntensityType>( integral ) );
  }

}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ForwardImageProjector3Dto2D<IntensityType>
::AfterThreadedGenerateData(void)
{
#if 0
  static int n = 0;
  char filename[256];

  // First cast the image from double to float

  typedef itk::Image< float, 3 > FloatOutputImageType;
  typedef itk::CastImageFilter< OutputImageType, FloatOutputImageType > CastFilterType;

  typename CastFilterType::Pointer  caster =  CastFilterType::New();

  caster->SetInput( this->GetOutput() );

  typedef itk::ImageFileWriter< FloatOutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  sprintf(filename, "/tmp/ForwardImageProjector3Dto2D_OUTPUT_%03d.gipl", ++n );
  writer->SetFileName( filename );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << "AfterThreadedGenerateData: " << filename << std::endl;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << filename << "; " << err << endl;
  }

  if (n >= 21)
    exit(1);
#endif


#if 0
  OutputImagePointer     outImage = this->GetOutput();
  ImageRegionIterator<OutputImageType> outputIterator;

  cout << endl << "DEBUG - Output of forward projection: " << endl;
 
  outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetLargestPossibleRegion());

  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) 
      cout << outputIterator.Get() << " ";

  cout << endl;
#endif
}

} // end namespace itk


#endif
