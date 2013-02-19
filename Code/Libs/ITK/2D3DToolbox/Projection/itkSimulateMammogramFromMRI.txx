/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSimulateMammogramFromMRI_txx
#define __itkSimulateMammogramFromMRI_txx

#include "itkSimulateMammogramFromMRI.h"

#include "itkLogHelper.h"

namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
SimulateMammogramFromMRI<IntensityType>
::SimulateMammogramFromMRI()
{

}


/* -----------------------------------------------------------------------
   PrintSelf(std::ostream&, Indent)
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
SimulateMammogramFromMRI<IntensityType>::
PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimulateMammogramFromMRI<IntensityType>
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

  sprintf(filename, "/tmp/SimulateMammogramFromMRI_INPUT_%03d.gipl", ++n );
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
SimulateMammogramFromMRI<IntensityType>
::GenerateData(void)
{
  // Perform multi-threaded execution by default

  if (this->m_FlagMultiThreadedExecution) {
    
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
	ray.IntegrateAboveThreshold(integral, this->m_Threshold);

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
SimulateMammogramFromMRI<IntensityType>
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
    ray.IntegrateAboveThreshold(integral, this->m_Threshold);

    outputIterator.Set( static_cast<IntensityType>( integral ) );
  }

}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimulateMammogramFromMRI<IntensityType>
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

  sprintf(filename, "/tmp/SimulateMammogramFromMRI_OUTPUT_%03d.gipl", ++n );
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
