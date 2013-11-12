/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCreateForwardBackwardProjectionMatrix_txx
#define __itkCreateForwardBackwardProjectionMatrix_txx

#include "itkCreateForwardBackwardProjectionMatrix.h"

#include <itkImageRegionIterator.h>
#include <itkProgressReporter.h>

#include <itkCastImageFilter.h>

#include <itkLogHelper.h>


namespace itk
{
/* -----------------------------------------------------------------------
     Constructor
     ----------------------------------------------------------------------- */

template <class IntensityType>
CreateForwardBackwardProjectionMatrix<IntensityType>
::CreateForwardBackwardProjectionMatrix()
{
  // Initialise the threshold above which intensities are integrated

  m_Threshold = 0.;

  // Multi-threaded execution is enabled by default

  m_FlagMultiThreadedExecution = false;

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
CreateForwardBackwardProjectionMatrix<IntensityType>::
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
CreateForwardBackwardProjectionMatrix<IntensityType>
::GenerateOutputInformation()
{
  OutputImageRegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize( m_OutputImageSize );

  OutputImagePointer outputPtr = this->GetOutput();
  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );  

  niftkitkDebugMacro(<<"Forward-projection output size: " << outputPtr->GetLargestPossibleRegion().GetSize());
}

/* -----------------------------------------------------------------------
     GenerateInputRequestedRegion()
     ----------------------------------------------------------------------- */

template< class IntensityType>
void
CreateForwardBackwardProjectionMatrix<IntensityType>
::GenerateInputRequestedRegion()
{
  // generate everything in the region of interest
  InputImagePointer  inputPtr = const_cast<InputImageType *> (this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  niftkitkDebugMacro(<<"Forward-projection input size: " << inputPtr->GetLargestPossibleRegion().GetSize());
}

/* -----------------------------------------------------------------------
     EnlargeOutputRequestedRegion(DataObject *)
     ----------------------------------------------------------------------- */

template< class IntensityType>
void
CreateForwardBackwardProjectionMatrix<IntensityType>
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
CreateForwardBackwardProjectionMatrix<IntensityType>
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

  sprintf(filename, "/tmp/CreateForwardBackwardProjectionMatrix_INPUT_%03d.gipl", ++n );
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
CreateForwardBackwardProjectionMatrix<IntensityType>
::GenerateData(void)
{
  // Perform multi-threaded execution by default

  if (m_FlagMultiThreadedExecution) {

	niftkitkDebugMacro(<<"Multi-threaded forward-projection");

    Superclass::GenerateData();

  }

  // Single-threaded execution

  else {

	niftkitkDebugMacro(<<"Single-threaded forward-projection");

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

    const InputImageSizeType &sizeOfInputImage  = inImage->GetLargestPossibleRegion().GetSize();
    const OutputImageSizeType &sizeOfOutputImage = outImage->GetLargestPossibleRegion().GetSize();

    // Create the ray object
    ray.SetImage( inImage );

    projMatrix = this->m_PerspectiveTransform->GetMatrix();
    projMatrix *= this->m_AffineTransform->GetFullAffineMatrix();

    ray.SetProjectionMatrix(projMatrix);

    // Iterate over pixels in the 2D projection (i.e. output) image
    outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetRequestedRegion());

    // Define a sparse matrix to store the forward projection matrix coefficients
    SparseMatrixType m_sparseForwardProjMatrix(sizeOfOutputImage[0]*sizeOfOutputImage[1],
					       sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);

    // Initialise the index of the intersection point
    double y, z = 0;
    const int* index;
				
    /*
				// This is used to normalise the projected intensities
				double pointSpace = 0.;
				double normCoef = 0.;
				InputImageSpacingType  inputImageSpacing  = inImage->GetSpacing();
				OutputImageSpacingType outputImageSpacing = outImage->GetSpacing();
				*/

    // Loop over each pixel in the projected 2D image, and from each pixel we cast a ray to the 3D volume
    // Iterate over pixels in the 2D projection (i.e. output) image
    int pixel2D = 0;
    for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
      {

	// Determine the coordinate of the output pixel
	outIndex = outputIterator.GetIndex();
	outImage->TransformIndexToPhysicalPoint(outIndex, outPoint);

	// Create a ray for this coordinate
	ray.SetRay(outPoint);

	integral = 0.;
	while (ray.NextPoint()) {

	  ray.GetBilinearCoefficients(y, z);
	  index = ray.GetRayIntersectionVoxelIndex();

	  integral += ray.GetCurrentIntensity();

	  /*
						// This is used to normalise the projected intensities
						pointSpace = ray.GetRayPointSpacing();
						normCoef = pointSpace*outputImageSpacing[0]*outputImageSpacing[1] / (inputImageSpacing[0]*inputImageSpacing[1]*inputImageSpacing[2]);
						*/
	
	  switch( ray.GetTraversalDirection() )
            {
	    case TRANSVERSE_IN_X:
	    {
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*index[1]
					+ index[0])
		= 1. - y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*(index[1]+1)
					+ index[0])
		= 1. - z + y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*index[1]
					+ (index[0]+1))
		= 1. - y + y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*(index[1]+1)
					+ (index[0]+1))
		= 1. + y + z - y*z;
	      break;
	    }
	    case TRANSVERSE_IN_Y:
	    {
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*index[1]
					+ index[0])
		= 1. - y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*(index[2]+1)
					+ sizeOfInputImage[1]*index[1]
					+ index[0])
		= 1. - z + y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*index[1]
					+ (index[0]+1))
		= 1. - y + y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*(index[2]+1)
					+ sizeOfInputImage[1]*index[1]
					+ (index[0]+1))
		= 1. + y + z - y*z;
	      break;
	    }
	    case TRANSVERSE_IN_Z: // Only this case makes the changes
	    {
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*index[1]
					+ index[0])
		= 1. - y*z; 
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*(index[1]+1)
					+ index[0])
		= 1. - y + y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*index[1]
					+ (index[0]+1))
		= 1. - z + y*z;
	      m_sparseForwardProjMatrix(pixel2D,
					sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
					+ sizeOfInputImage[1]*(index[1]+1)
					+ (index[0]+1))
		= 1. + y + z - y*z;
	      break;
	      /* case TRANSVERSE_IN_Z: // Only this case makes the changes
                {
                  m_sparseForwardProjMatrix(pixel2D,
                      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
                      + sizeOfInputImage[1]*index[1]
                      + index[0])
                    = (1. - y*z)*normCoef; 
                  m_sparseForwardProjMatrix(pixel2D,
                      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
                      + sizeOfInputImage[1]*(index[1]+1)
                      + index[0])
                    = (1. - y + y*z)*normCoef;
                  m_sparseForwardProjMatrix(pixel2D,
                      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
                      + sizeOfInputImage[1]*index[1]
                      + (index[0]+1))
                    = (1. - z + y*z)*normCoef;
                  m_sparseForwardProjMatrix(pixel2D,
                      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
                      + sizeOfInputImage[1]*(index[1]+1)
                      + (index[0]+1))
                    = (1. + y + z - y*z)*normCoef;
                  break; */
	    }
            }
	}

	// std::cerr << "Deal with the iteration number " << pixel2D << std::endl;
	// std::cerr << " " << std::endl;
	pixel2D++;		

	// Calculate the ray casting integration
	outputIterator.Set( static_cast<IntensityType>( integral ) );
      }

    // Covert the input image into the vnl vector form
    typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ConstIteratorType;

    VectorType inputImageVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);	
    // inputImageVector.fill(1.);
    // std::ofstream inputImageVectorFile("inputImageVector.txt");
    // inputImageVectorFile << inputImageVector << " ";


    ConstIteratorType inputIterator( inImage, inImage->GetLargestPossibleRegion() );			

    unsigned int voxel3D = 0;
    InputImagePixelType voxelValue;
    for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator)
      {

	voxelValue = inputIterator.Get();
	inputImageVector.put(voxel3D, (double) voxelValue);

	voxel3D++;	

      }
    // std::cerr << "vexel3D is " << voxel3D << std::endl;
    // std::cerr << "The max value of the vector is " << inputImageVector.max_value() << std::endl;
    // std::cerr << " " << std::endl;
    // std::cerr << "The size of the input vector is " << inputImageVector.size() << std::endl;
    // std::cerr << " " << std::endl;
    std::ofstream inputImageVectorFile("inputImageVector.txt");
    inputImageVectorFile << inputImageVector << " ";


    // SparseMatrixType sparseTest(10000, 125000);
    // for (int rowNumberSp = 10; rowNumberSp <1299; ++rowNumberSp)
    //  	for (int colmnNumberSp = 1000; colmnNumberSp <1008; ++colmnNumberSp)
    //  		sparseTest(rowNumberSp, colmnNumberSp) = 0.98;


    // Calculate the matrix/vector multiplication in order to get the forward projection
    assert (!inputImageVector.is_zero());
    VectorType outputImageVector(sizeOfOutputImage[0]*sizeOfOutputImage[1]);
    outputImageVector.fill(0.);

    // VectorType copyInputImageVector = inputImageVector; // ?? Without adding this will end with run-time segmentation error when single threaded
    m_sparseForwardProjMatrix.mult(inputImageVector, outputImageVector);
    // sparseTest.mult(inputImageVector, outputImageVector);

    // std::cerr << "m_sparseForwardProjMatrix is " << m_sparseForwardProjMatrix.rows() << " by " << m_sparseForwardProjMatrix.cols() << std::endl;
    // std::cerr << "inputImageVector is " << inputImageVector.size() << " by 1" << std::endl;
    // std::cerr << "outputImageVector is " << outputImageVector.size() << " by 1" << std::endl;

    std::ofstream vectorFile("vectorFile.txt", std::ios::out | std::ios::app | std::ios::binary) ;
    vectorFile << outputImageVector << " ";

    // Calculate the transpose of the forward projection matrix in order to obtain the backward projection
    SparseMatrixType m_sparseBackwardProjMatrix(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2],
						sizeOfOutputImage[0]*sizeOfOutputImage[1]);

    int rowIndex = 0;
    int colIndex = 0;
    m_sparseForwardProjMatrix.reset();
    while ( m_sparseForwardProjMatrix.next() )
      {
	rowIndex = m_sparseForwardProjMatrix.getrow();
	colIndex = m_sparseForwardProjMatrix.getcolumn();
	m_sparseBackwardProjMatrix(colIndex, rowIndex) = m_sparseForwardProjMatrix.value();
      }

    // Calculate the transpose of the forward projection matrix in order to obtain the backward projection
    assert (!outputImageVector.is_zero());
    VectorType outputBackProjImageVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);
    outputBackProjImageVector.fill(0.);

    m_sparseBackwardProjMatrix.mult(outputImageVector, outputBackProjImageVector);
    // sparseTest.mult(inputImageVector, outputImageVector);

    // std::cerr << "m_sparseForwardProjMatrix is " << m_sparseBackwardProjMatrix.rows() << " by " << m_sparseForwardProjMatrix.cols() << std::endl;
    // std::cerr << "inputImageVector is " << outputImageVector.size() << " by 1" << std::endl;
    // std::cerr << "outputImageVector is " << outputBackProjImageVector.size() << " by 1" << std::endl;

    std::ofstream vectorBackProjFile("vectorBackProjFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    // std::ofstream vectorBackProjFile("vectorBackProjFile.txt");
    vectorBackProjFile << outputBackProjImageVector << " ";	

    /*
        // Print out the number of none-zero rows
        std::ofstream noneZeroRowsFile("noneZeroRows.txt");
        unsigned int numberMatrixIter = 0;
        while ( numberMatrixIter < m_sparseForwardProjMatrix.rows() )
        {				
        if (!(m_sparseForwardProjMatrix.empty_row(numberMatrixIter)))
        {
        // std::cerr << "The row number " << numberMatrixIter << " is not empty." << std::endl;
        // std::cerr << " " << std::endl;
        noneZeroRowsFile << "\nThe row number " << numberMatrixIter << " is not empty.\n";
        }

        numberMatrixIter++;
        }

        // Define the vectors to extract columns of the sparse matrix
        VectorType extractColumnsVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);
        VectorType sparseMatrixColumnsVector(sizeOfOutputImage[0]*sizeOfOutputImage[1]);

        std::ofstream sparseMatrixFile("sparseMatrix.txt");
        unsigned int columnNumberMatrixIter = 0;
        while ( columnNumberMatrixIter < m_sparseForwardProjMatrix.cols() ) // Too long to get the whole matrix to be saved
        // while ( columnNumberMatrixIter < 100)
        {
        // Output the sparse forward projection matrix as a .txt file
        extractColumnsVector.fill(0.);
        sparseMatrixColumnsVector.fill(0.);
        assert ( (extractColumnsVector.is_zero()) && (sparseMatrixColumnsVector.is_zero()) );
        extractColumnsVector.put(columnNumberMatrixIter, 1.);
        assert ( !extractColumnsVector.is_zero() );

        m_sparseForwardProjMatrix.mult(inputImageVector, sparseMatrixColumnsVector);
        sparseMatrixFile << sparseMatrixColumnsVector << " ";

        columnNumberMatrixIter++;
        }
         */

    /*
        // Construct the full matrix
        typedef vnl_matrix<double>           FullMatrixType;
        FullMatrixType fullForwardProjMatrix(sizeOfOutputImage[0]*sizeOfOutputImage[1],
        sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2], 0);
         */

    /*
        // Print out the non-zero elements
        std::ofstream noneZeroElementsFile("noneZeroElements.txt");
        while ( m_sparseForwardProjMatrix.next() )
        {
        // fullForwardProjMatrix( m_sparseForwardProjMatrix.getrow(), m_sparseForwardProjMatrix.getcolumn() ) = m_sparseForwardProjMatrix.value();

        std::cerr << " " << std::endl;
        std::cerr << "The value of [" << m_sparseForwardProjMatrix.getrow() << ", " << m_sparseForwardProjMatrix.getcolumn()
        << "] is: " << m_sparseForwardProjMatrix.value() << std::endl;
        std::cerr << " " << std::endl;
        noneZeroElementsFile << "\nThe value of [" << m_sparseForwardProjMatrix.getrow() << ", " << m_sparseForwardProjMatrix.getcolumn()
        << "] is: " << m_sparseForwardProjMatrix.value() << "\n";
        }
         */

  }
}


/* -----------------------------------------------------------------------
     ThreadedGenerateData(const OutputImageRegionType&, int)
     ----------------------------------------------------------------------- */

template< class IntensityType>
void
CreateForwardBackwardProjectionMatrix<IntensityType>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
           ThreadIdType threadId)
{

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


  // Allocate output
  InputImageConstPointer inImage  = this->GetInput();
  OutputImagePointer     outImage = this->GetOutput();

  const InputImageSizeType &sizeOfInputImage  = inImage->GetLargestPossibleRegion().GetSize();
  const OutputImageSizeType &sizeOfOutputImage = outImage->GetLargestPossibleRegion().GetSize();

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

  // Get the region size of each thread
  OutputImageRegionType outputImageRegion = outputIterator.GetRegion();
  const OutputImageSizeType &sizeOfOutputRegion = outputImageRegion.GetSize();
  // std::cerr << "sizeOfOutputRegion is " << sizeOfOutputRegion[0] << " and " << sizeOfOutputRegion[1] << std::endl;
  // std::cerr << " " << std::endl;

  // Define a sparse matrix to store the forward projection matrix coefficients
  SparseMatrixType m_sparseForwardProjMatrix(sizeOfOutputImage[0]*sizeOfOutputImage[1],
					     sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);

  // std::cerr << " " << std::endl;
  // std::cerr << "The number of rows of the forward projection matrix is " << m_sparseForwardProjMatrix.rows()
  //   << ", and the number of column is " << m_sparseForwardProjMatrix.columns() << "." << std::endl;
  // std::cerr << " " << std::endl;

  // Initialise the index of the intersection point
  double y, z = 0.;
  const int* index;

  // Determine the index for each threaded bulk of data
  outputIterator.GoToBegin();
  OutputImageIndexType outIndexStart = outputIterator.GetIndex();
  int pixel2D = sizeOfOutputImage[1]*outIndexStart[1] + outIndexStart[0];

  // Loop over each pixel in the projected 2D image, and from each pixel we cast a ray to the 3D volume
  // Iterate over pixels in the 2D projection (i.e. output) image
  // int pixel2D = 0;
  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
    {
      // Determine the coordinate of the output pixel
      outIndex = outputIterator.GetIndex();
      outImage->TransformIndexToPhysicalPoint(outIndex, outPoint);



      // Create a ray for this coordinate
      ray.SetRay(outPoint);

      integral = 0.;

      while (ray.NextPoint()) {

	ray.GetBilinearCoefficients(y, z);
	index = ray.GetRayIntersectionVoxelIndex();
	// std::cerr << "The indices of intersection are [" << index[2] << ", " << index[1] << ", " << index[0] << "]." << std::endl;
	// std::cerr << " " << std::endl;

	// if ( 0 && (ray.GetTraversalDirection()) != 0)
	// {
	//   std::cerr << "The traversal direction is " << ray.GetTraversalDirection() << std::endl;
	//   std::cerr << " " << std::endl;
	// }

	integral += ray.GetCurrentIntensity();

	switch( ray.GetTraversalDirection() )
          {
	  case TRANSVERSE_IN_X:
	  {
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*index[1]
				      + index[0])
	      = 1. - y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*(index[1]+1)
				      + index[0])
	      = 1. - z + y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*index[1]
				      + (index[0]+1))
	      = 1. - y + y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*(index[1]+1)
				      + (index[0]+1))
	      = 1. + y + z - y*z;
	    break;
	  }
	  case TRANSVERSE_IN_Y:
	  {
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*index[1]
				      + index[0])
	      = 1. - y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*(index[2]+1)
				      + sizeOfInputImage[1]*index[1]
				      + index[0])
	      = 1. - z + y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*index[1]
				      + (index[0]+1))
	      = 1. - y + y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*(index[2]+1)
				      + sizeOfInputImage[1]*index[1]
				      + (index[0]+1))
	      = 1. + y + z - y*z;
	    break;
	  }
	  case TRANSVERSE_IN_Z: // Only this case makes the changes
	  {
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*index[1]
				      + index[0])
	      = 1. - y*z; 
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*(index[1]+1)
				      + index[0])
	      = 1. - y + y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*index[1]
				      + (index[0]+1))
	      = 1. - z + y*z;
	    m_sparseForwardProjMatrix(pixel2D,
				      sizeOfInputImage[2]*sizeOfInputImage[1]*index[2]
				      + sizeOfInputImage[1]*(index[1]+1)
				      + (index[0]+1))
	      = 1. + y + z - y*z;
	    break;
	  }
          }
      }

      // std::ofstream pixelNumberFile("pixelNumberFile.txt");
      // pixelNumberFile << "Deal with the iteration number " << pixel2D << '\n' << std::endl;
      pixel2D++;

      // Calculate the ray casting integration
      outputIterator.Set( static_cast<IntensityType>( integral ) );

    }

  /*
      // Test sparse matrix multiply with vector
      SparseMatrixType sparseTest(10000, 125000);
      for (int rowNumberSp = 10; rowNumberSp <1299; ++rowNumberSp)
      for (int colmnNumberSp = 1000; colmnNumberSp <1008; ++colmnNumberSp)
      sparseTest(rowNumberSp, colmnNumberSp) = 0.98;
      std::cerr << "sparseTest is " << sparseTest.rows() << " by " << sparseTest.cols() << std::endl;
      VectorType vectorTest(125000);
      VectorType vectorResult(10000);
      vectorTest.fill(.1);
      vectorResult.fill(0.);
      std::cerr << "vectorTest is " << vectorTest.size() << " by 1 before mult." << std::endl;
      std::cerr << "vectorResult is " << vectorResult.size() << " by 1 before mult." << std::endl;
      sparseTest.mult(vectorTest, vectorResult);
      std::cerr << "vectorTest is " << vectorTest.size() << " by 1 after mult." << std::endl;
      std::cerr << "vectorResult is " << vectorResult.size() << " by 1 after mult." << std::endl;	
      std::ofstream vectorResultFile("vectorResult.txt");
      vectorResultFile << vectorResult << " ";
       */

  // Covert the input image into the vnl vector form
  typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ConstIteratorType;

  VectorType inputImageVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);	
  // inputImageVector.fill(1.);
  // std::ofstream inputImageVectorFile("inputImageVector.txt");
  // inputImageVectorFile << inputImageVector << " ";


  ConstIteratorType inputIterator( inImage, inImage->GetRequestedRegion() );			

  unsigned int voxel3D = 0;
  InputImagePixelType voxelValue;
  for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator)
    {

      voxelValue = inputIterator.Get();
      inputImageVector.put(voxel3D, (double) voxelValue);

      voxel3D++;	

    }
  // std::cerr << "vexel3D is " << voxel3D << std::endl;
  // std::cerr << "The max value of the vector is " << inputImageVector.max_value() << std::endl;
  // std::cerr << " " << std::endl;
  // std::cerr << "The size of the input vector is " << inputImageVector.size() << std::endl;
  // std::cerr << " " << std::endl;
  std::ofstream inputImageVectorFile("inputImageVector.txt");
  inputImageVectorFile << inputImageVector << " ";


  // SparseMatrixType sparseTest(10000, 125000);
  // for (int rowNumberSp = 10; rowNumberSp <1299; ++rowNumberSp)
  //  	for (int colmnNumberSp = 1000; colmnNumberSp <1008; ++colmnNumberSp)
  //  		sparseTest(rowNumberSp, colmnNumberSp) = 0.98;


  // Calculate the matrix/vector multiplication in order to get the forward projection
  assert (!inputImageVector.is_zero());
  VectorType outputImageVector(sizeOfOutputImage[0]*sizeOfOutputImage[1]);
  outputImageVector.fill(0.);

  m_sparseForwardProjMatrix.mult(inputImageVector, outputImageVector);
  // sparseTest.mult(inputImageVector, outputImageVector);

  // std::cerr << "m_sparseForwardProjMatrix is " << m_sparseForwardProjMatrix.rows() << " by " << m_sparseForwardProjMatrix.cols() << std::endl;
  // std::cerr << "inputImageVector is " << inputImageVector.size() << " by 1" << std::endl;
  // std::cerr << "outputImageVector is " << outputImageVector.size() << " by 1" << std::endl;

  // std::ofstream vectorFile("vectorFile.txt");
  // vectorFile << outputImageVector << " ";

  // Create a vector to store the values of the output vector per thread
  VectorType m_outputVectorPerThread(sizeOfOutputRegion[0]*sizeOfOutputRegion[1]);
  for (unsigned int vectorNum = 0; vectorNum < sizeOfOutputRegion[0]*sizeOfOutputRegion[1]; vectorNum++)
    m_outputVectorPerThread.put(vectorNum, outputImageVector.get(threadId*sizeOfOutputRegion[0]*sizeOfOutputRegion[1]+vectorNum));	

  // Map the vector per thread into the map with thread ID 
  m_outputVectorAllThread[threadId] = m_outputVectorPerThread;

/*
			// Calculate the transpose of the forward projection matrix in order to obtain the backward projection
      SparseMatrixType m_sparseBackwardProjMatrix(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2],
				sizeOfOutputImage[0]*sizeOfOutputImage[1]);

			int rowIndex = 0;
      int colIndex = 0;
			m_sparseForwardProjMatrix.reset();
			while ( m_sparseForwardProjMatrix.next() )
			{
				rowIndex = m_sparseForwardProjMatrix.getrow();
				colIndex = m_sparseForwardProjMatrix.getcolumn();
				m_sparseBackwardProjMatrix(colIndex, rowIndex) = m_sparseForwardProjMatrix.value();
			}

      // assert (!outputImageVector.is_zero());
      VectorType outputBackProjImageVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);
      outputBackProjImageVector.fill(0.);

      m_sparseBackwardProjMatrix.mult(outputImageVector, outputBackProjImageVector);

      // Create a vector to store the values of the output vector per thread
      VectorType m_outputVectorBackProjPerThread(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);
      // for (unsigned int vectorNum = 0; vectorNum < sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]; vectorNum++)
      //  m_outputVectorBackProjPerThread.put(vectorNum, 
			//		outputBackProjImageVector.get(threadId*sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]+vectorNum));	

      // Map the vector per thread into the map with thread ID 
			m_outputVectorBackProjAllThread[threadId] = m_outputVectorBackProjPerThread;
*/

  /*
      // Print out the number of none-zero rows
      std::ofstream noneZeroRowsFile("noneZeroRows.txt");
      unsigned int numberMatrixIter = 0;
      while ( numberMatrixIter < m_sparseForwardProjMatrix.rows() )
      {				
      if (!(m_sparseForwardProjMatrix.empty_row(numberMatrixIter)))
      {
      // std::cerr << "The row number " << numberMatrixIter << " is not empty." << std::endl;
      // std::cerr << " " << std::endl;
      noneZeroRowsFile << "\nThe row number " << numberMatrixIter << " is not empty.\n";
      }

      numberMatrixIter++;
      }
       */

  /*
      // Define the vectors to extract columns of the sparse matrix
      VectorType extractColumnsVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);
      VectorType sparseMatrixColumnsVector(sizeOfOutputImage[0]*sizeOfOutputImage[1]);

      std::ofstream sparseMatrixFile("sparseMatrix.txt");
      unsigned int columnNumberMatrixIter = 0;
      while ( columnNumberMatrixIter < m_sparseForwardProjMatrix.cols() ) // Too long to get the whole matrix to be saved
      // while ( columnNumberMatrixIter < 100)
      {
      // Output the sparse forward projection matrix as a .txt file
      extractColumnsVector.fill(0.);
      sparseMatrixColumnsVector.fill(0.);
      assert ( (extractColumnsVector.is_zero()) && (sparseMatrixColumnsVector.is_zero()) );
      extractColumnsVector.put(columnNumberMatrixIter, 1.);
      assert ( !extractColumnsVector.is_zero() );

      m_sparseForwardProjMatrix.mult(inputImageVector, sparseMatrixColumnsVector);
      sparseMatrixFile << sparseMatrixColumnsVector << " "; 

      columnNumberMatrixIter++;
      }
       */

  /*
      // Construct the full matrix
      typedef vnl_matrix<double>           FullMatrixType;
      FullMatrixType fullForwardProjMatrix(sizeOfOutputImage[0]*sizeOfOutputImage[1],
      sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2], 0);
       */

  /*
      // Print out the non-zero elements
      std::ofstream noneZeroElementsFile("noneZeroElements.txt");
      while ( m_sparseForwardProjMatrix.next() )
      {
      // fullForwardProjMatrix( m_sparseForwardProjMatrix.getrow(), m_sparseForwardProjMatrix.getcolumn() ) = m_sparseForwardProjMatrix.value();

      std::cerr << " " << std::endl;
      std::cerr << "The value of [" << m_sparseForwardProjMatrix.getrow() << ", " << m_sparseForwardProjMatrix.getcolumn()
      << "] is: " << m_sparseForwardProjMatrix.value() << std::endl;
      std::cerr << " " << std::endl;
      noneZeroElementsFile << "\nThe value of [" << m_sparseForwardProjMatrix.getrow() << ", " << m_sparseForwardProjMatrix.getcolumn()
      << "] is: " << m_sparseForwardProjMatrix.value() << "\n";
      }
       */

  // Call a method that can be overridden by a subclass to perform
  // some calculations after all the threads have completed
  this->AfterThreadedGenerateData();
}


/* -----------------------------------------------------------------------
     AfterThreadedGenerateData()
     ----------------------------------------------------------------------- */

template< class IntensityType>
void
CreateForwardBackwardProjectionMatrix<IntensityType>
::AfterThreadedGenerateData(void)
{

  // Backward projection matrix is not working for multi-thread program !!!

  // Store the output vector after all the threads been processed 
  int numberOfThreads = this->GetNumberOfThreads();

  std::ofstream vectorFile("vectorFile.txt");
  for ( int threadsIter = 0; threadsIter < numberOfThreads; threadsIter++ )
    {
      // std::cerr << " " << std::endl;
      // std::cerr << "The size of the data at thread " << threadsIter << " is " << m_outputVectorAllThread[threadsIter].size() << std::endl;
      // std::cerr << " " << std::endl;

      // Put a space at last to separate vectors produced by different threads
      if (m_outputVectorAllThread[threadsIter].size() > 0 )
	vectorFile << m_outputVectorAllThread[threadsIter] << " ";

      // Put a space at last to separate vectors produced by different threads
      // if (m_outputVectorBackProjAllThread[threadsIter].size() > 0 )
      //  vectorBackProjFile << m_outputVectorBackProjAllThread[threadsIter] << " ";
    }

/*
			static int suffixNum = 0;
			char outputVectorFileName[256];

			sprintf(outputVectorFileName, "vectorFile_%03d.txt", suffixNum++ );

      // Store the output vector after all the threads been processed 
      int numberOfThreads = this->GetNumberOfThreads();

			std::ofstream vectorFile("vectorFile.txt");
      // std::ofstream vectorFile("vectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
			// std::ofstream vectorBackProjFile("vectorBackProjFile.txt", std::ios::out | std::ios::app | std::ios::binary);
      for ( int threadsIter = 0; threadsIter < numberOfThreads; threadsIter++ )
      {
        // std::cerr << " " << std::endl;
        // std::cerr << "The size of the data at thread " << threadsIter << " is " << m_outputVectorAllThread[threadsIter].size() << std::endl;
        // std::cerr << " " << std::endl;

        // Put a space at last to separate vectors produced by different threads
        if (m_outputVectorAllThread[threadsIter].size() > 0 )
          vectorFile << m_outputVectorAllThread[threadsIter] << " ";

        // Put a space at last to separate vectors produced by different threads
        // if (m_outputVectorBackProjAllThread[threadsIter].size() > 0 )
        //  vectorBackProjFile << m_outputVectorBackProjAllThread[threadsIter] << " ";
      }
*/

/*
			static int suffixNum = 0;
			int numberOfThreads = this->GetNumberOfThreads();

			std::ofstream vectorFileTemp("vectorFileTemp.txt");
      for ( int threadsIter = 0; threadsIter < numberOfThreads; threadsIter++ )
      {
        // Put a space at last to separate vectors produced by different threads
        if (m_outputVectorAllThread[threadsIter].size() > 0 )
          vectorFileTemp << m_outputVectorAllThread[threadsIter] << " ";
      }

			if (	suffixNum == numberOfThreads )
			{
				std::string lineTemp;
				std::ofstream vectorFile("vectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);

				std::ifstream vectorFileTempRead("vectorFileTemp.txt");
				while( getline(vectorFileTempRead, lineTemp) )
          vectorFile << lineTemp << " "; 
			}

			suffixNum++;
*/


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

  sprintf(filename, "/tmp/CreateForwardBackwardProjectionMatrix_OUTPUT_%03d.gipl", ++n );
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

