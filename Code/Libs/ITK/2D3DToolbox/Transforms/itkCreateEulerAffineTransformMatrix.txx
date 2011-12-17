/*=============================================================================

NifTK: An image processing toolkit jointly developed by the
Dementia Research Centre, and the Centre For Medical Image Computing
at University College London.

See:        http://dementia.ion.ucl.ac.uk/
http://cmic.cs.ucl.ac.uk/
http://www.ucl.ac.uk/

Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
Revision          : $Revision: 3326 $
Last modified by  : $Author: jhh, gy $

Original author   : j.hipwell@ucl.ac.uk

Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

============================================================================*/

#ifndef __itkCreateEulerAffineTransformMatrix_txx
#define __itkCreateEulerAffineTransformMatrix_txx

#include "itkCreateEulerAffineTransformMatrix.h"

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
    CreateEulerAffineTransformMatrix<IntensityType>
    ::CreateEulerAffineTransformMatrix()
    {
      // Multi-threaded execution is enabled by default

      m_FlagMultiThreadedExecution = false;

      // Set default values for the output image size

      m_OutputImageSize[0]  = 100;  // size along X
      m_OutputImageSize[1]  = 100;  // size along Y
      m_OutputImageSize[2]  = 100;  // size along Z

      // Set default values for the output image resolution

      m_OutputImageSpacing[0]  = 1;  // resolution along X axis
      m_OutputImageSpacing[1]  = 1;  // resolution along Y axis
      m_OutputImageSpacing[2]  = 1;  // resolution along Z axis

      // Set default values for the output image origin

      m_OutputImageOrigin[0]  = 0.;  // origin in X
      m_OutputImageOrigin[1]  = 0.;  // origin in Y
      m_OutputImageOrigin[2]  = 0.;  // origin in Z

      OutputImagePointer      outputPtr = this->GetOutput();

      outputPtr->SetSpacing(m_OutputImageSpacing);
      outputPtr->SetOrigin(m_OutputImageOrigin);
    }


  /* -----------------------------------------------------------------------
     PrintSelf(std::ostream&, Indent)
     ----------------------------------------------------------------------- */

  template <class IntensityType>
    void
    CreateEulerAffineTransformMatrix<IntensityType>::
    PrintSelf(std::ostream& os, Indent indent) const
    {
      Superclass::PrintSelf(os,indent);

      os << indent << "Output image size: " << m_OutputImageSize << std::endl;
      os << indent << "Output image spacing: " << m_OutputImageSpacing << std::endl;
      os << indent << "Output image origin: " << m_OutputImageOrigin << std::endl;

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
    CreateEulerAffineTransformMatrix<IntensityType>
    ::GenerateOutputInformation()
    {
      OutputImageRegionType outputLargestPossibleRegion;
      outputLargestPossibleRegion.SetSize( m_OutputImageSize );

      OutputImagePointer outputPtr = this->GetOutput();
      outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );  

      niftkitkDebugMacro(<<"Affine transformed output size: " << outputPtr->GetLargestPossibleRegion().GetSize());
    }

  /* -----------------------------------------------------------------------
     GenerateInputRequestedRegion()
     ----------------------------------------------------------------------- */

  template< class IntensityType>
    void
    CreateEulerAffineTransformMatrix<IntensityType>
    ::GenerateInputRequestedRegion()
    {
      // generate everything in the region of interest
      InputImagePointer  inputPtr = const_cast<InputImageType *> (this->GetInput());
      inputPtr->SetRequestedRegionToLargestPossibleRegion();

      niftkitkDebugMacro(<<"Affine transformation input size: " << inputPtr->GetLargestPossibleRegion().GetSize());
    }

  /* -----------------------------------------------------------------------
     EnlargeOutputRequestedRegion(DataObject *)
     ----------------------------------------------------------------------- */

  template< class IntensityType>
    void
    CreateEulerAffineTransformMatrix<IntensityType>
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
    CreateEulerAffineTransformMatrix<IntensityType>
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

      sprintf(filename, "/tmp/CreateEulerAffineTransformMatrix_INPUT_%03d.gipl", ++n );
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
    CreateEulerAffineTransformMatrix<IntensityType>
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

        // Call a method that can be overridden by a subclass to perform
        // some calculations prior to splitting the main computations into
        // separate threads
        this->BeforeThreadedGenerateData();

        OutputImageIndexType outIndex;

        ImageRegionIterator<OutputImageType> outputIterator;

        // Call a method that can be overriden by a subclass to allocate
        // memory for the filter's outputs
        this->AllocateOutputs();

        InputImageConstPointer inImage  = this->GetInput();
        OutputImagePointer     outImage = this->GetOutput();

        OutputImageSpacingType res2D = outImage->GetSpacing();

				const InputImageSizeType &sizeOfInputImage  = inImage->GetLargestPossibleRegion().GetSize();
        const OutputImageSizeType &sizeOfOutputImage = outImage->GetLargestPossibleRegion().GetSize();

        // Define the affine core matrix
        FullMatrixType m_affineCoreMatrix(4, 4);
        FullMatrixType m_affineCoreMatrixInverse(4, 4);
        VectorType m_inputCoordinateVector(4);
        VectorType m_outputCoordinateVector(4);
        Matrix<double, 4, 4> affineCoreMatrix;
        Matrix<double, 4, 4> affineCoreMatrixInverse;

        // Define a sparse matrix to store the affine transformation matrix coefficients
        SparseMatrixType m_sparseAffineTransformMatrix(sizeOfOutputImage[0]*sizeOfOutputImage[1]*sizeOfOutputImage[2],
            sizeOfOutputImage[0]*sizeOfOutputImage[1]*sizeOfOutputImage[2]);

        affineCoreMatrix = this->m_AffineTransform->GetFullAffineMatrix();
        m_affineCoreMatrix = affineCoreMatrix.GetVnlMatrix();
        std::cerr << "The core affine transformation matrix is:" << std::endl << m_AffineTransform->GetFullAffineMatrix() << std::endl; 
        m_AffineTransform->InvertTransformationMatrix(); 
        affineCoreMatrixInverse = m_AffineTransform->GetFullAffineMatrix();
        m_affineCoreMatrixInverse = affineCoreMatrixInverse.GetVnlMatrix();
        std::cerr << "The inverted core affine transformation matrix is:" << std::endl << m_AffineTransform->GetFullAffineMatrix() << std::endl;

        // Iterate over index in the 3D affine transformed (i.e. output) image

        outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetRequestedRegion());

        std::ofstream invTransCoorVectorFile("invTransCoorVectorFile.txt");
        std::ofstream originalCoorVectorFile("originalCoorVectorFile.txt");
        int voxelNum = 0;
        double xCoorCoef = 0.;
        double yCoorCoef = 0.;
        double leftBottomXCoorinate = 0.;
        double leftBottomYCoorinate = 0.;
        double leftBottomZCoorinate = 0.;
        int intLeftBottomXCoorinate = 0;
        int intLeftBottomYCoorinate = 0;
        int intLeftBottomZCoorinate = 0;
        for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) 
        {

          outIndex = outputIterator.GetIndex();
          m_outputCoordinateVector.put(0, outIndex[2]);
          m_outputCoordinateVector.put(1, outIndex[1]);
          m_outputCoordinateVector.put(2, outIndex[0]);
          m_outputCoordinateVector.put(3, 1);

          originalCoorVectorFile << std::endl << "The original coordinates of voxel number " << voxelNum<< " are:" 
            << std::endl << m_outputCoordinateVector[0] << " " << m_outputCoordinateVector[1] << " " 
            << m_outputCoordinateVector[2] << " " << m_outputCoordinateVector[3] << std::endl;

          m_inputCoordinateVector = m_affineCoreMatrixInverse*m_outputCoordinateVector;
          invTransCoorVectorFile << std::endl << "The inverse transformed coordinates of voxel number " << voxelNum<< " are:" 
            << std::endl << m_inputCoordinateVector[0] << " " << m_inputCoordinateVector[1] << " " 
            << m_inputCoordinateVector[2] << " " << m_inputCoordinateVector[3] << std::endl;

          leftBottomXCoorinate = vcl_floor(m_inputCoordinateVector[0]);
          leftBottomYCoorinate = vcl_floor(m_inputCoordinateVector[1]);
          leftBottomZCoorinate = vcl_floor(m_inputCoordinateVector[2]);
          intLeftBottomXCoorinate = (int) leftBottomXCoorinate;
          intLeftBottomYCoorinate = (int) leftBottomYCoorinate;
          intLeftBottomZCoorinate = (int) leftBottomZCoorinate;

          // xCoorCoef = vcl_abs(m_outputCoordinateVector[0] - leftBottomXCoorinate);
          // yCoorCoef = vcl_abs(m_outputCoordinateVector[1] - leftBottomYCoorinate);
          xCoorCoef = vcl_abs(m_inputCoordinateVector[2] - leftBottomXCoorinate);
          yCoorCoef = vcl_abs(m_inputCoordinateVector[1] - leftBottomYCoorinate);

          m_sparseAffineTransformMatrix(voxelNum,
              sizeOfOutputImage[2]*sizeOfOutputImage[1]*intLeftBottomZCoorinate
              + sizeOfOutputImage[1]*intLeftBottomYCoorinate
              + intLeftBottomXCoorinate)
            = 1. - xCoorCoef*yCoorCoef;

          m_sparseAffineTransformMatrix(voxelNum,
              sizeOfOutputImage[2]*sizeOfOutputImage[1]*intLeftBottomZCoorinate
              + sizeOfOutputImage[1]*(intLeftBottomYCoorinate+1)
              + intLeftBottomXCoorinate)
            = 1. - xCoorCoef + xCoorCoef*yCoorCoef; 

          m_sparseAffineTransformMatrix(voxelNum,
              sizeOfOutputImage[2]*sizeOfOutputImage[1]*intLeftBottomZCoorinate
              + sizeOfOutputImage[1]*intLeftBottomYCoorinate
              + (intLeftBottomXCoorinate+1))
            = 1. - yCoorCoef + xCoorCoef*yCoorCoef; 

          m_sparseAffineTransformMatrix(voxelNum,
              sizeOfOutputImage[2]*sizeOfOutputImage[1]*intLeftBottomZCoorinate
              + sizeOfOutputImage[1]*(intLeftBottomYCoorinate+1)
              + (intLeftBottomXCoorinate+1))
            = 1. + xCoorCoef + yCoorCoef - xCoorCoef*yCoorCoef;  

          voxelNum++;
        }

        // Covert the input image into the vnl vector form
        typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ConstIteratorType;
        ConstIteratorType inputIterator( inImage, inImage->GetLargestPossibleRegion() );	

        VectorType inputImageVector(sizeOfInputImage[0]*sizeOfInputImage[1]*sizeOfInputImage[2]);	
		
        unsigned int voxel3D = 0;
        InputImagePixelType voxelValue;
        for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator)
        {

          voxelValue = inputIterator.Get();
          inputImageVector.put(voxel3D, (double) voxelValue);

          voxel3D++;	

        }

        std::ofstream inputImageVectorFile("inputImageVector.txt");
        inputImageVectorFile << inputImageVector << " ";

        // Calculate the matrix/vector multiplication in order to get the forward projection
        assert (!inputImageVector.is_zero());
        VectorType outputImageVector(sizeOfOutputImage[0]*sizeOfOutputImage[1]*sizeOfOutputImage[2]);
        outputImageVector.fill(0.);

        m_sparseAffineTransformMatrix.mult(inputImageVector, outputImageVector);

        std::ofstream vectorFile("vectorFile.txt", std::ios::out | std::ios::app | std::ios::binary) ;
        vectorFile << outputImageVector << " ";

  			// Print out the non-zero entries

				m_sparseAffineTransformMatrix.reset();
				std::ofstream sparseAffineMatrixFile("sparseAffineMatrix.txt");
				sparseAffineMatrixFile << std::endl << "The non-zero entries of the affine matrix are: " << std::endl;

				unsigned int rowIndex = 0;
  			unsigned int colIndex = 0;

				while ( m_sparseAffineTransformMatrix.next() )
				{
					rowIndex = m_sparseAffineTransformMatrix.getrow();
					colIndex = m_sparseAffineTransformMatrix.getcolumn();
					
					if ( (rowIndex < m_sparseAffineTransformMatrix.rows()) && (colIndex < m_sparseAffineTransformMatrix.cols()) )	
							sparseAffineMatrixFile << std::endl << "Row " << rowIndex << " and column " << colIndex << " is: " << 
							m_sparseAffineTransformMatrix.value() << std::endl;
				} 

				// Calculate the transpose of the affine transformation matrix
        SparseMatrixType m_sparseTransposeAffineTransformMatrix(sizeOfOutputImage[0]*sizeOfOutputImage[1]*sizeOfOutputImage[2],
					sizeOfOutputImage[0]*sizeOfOutputImage[1]*sizeOfOutputImage[2]);

				// unsigned int rowIndex = 0;
        // unsigned int colIndex = 0;
				m_sparseAffineTransformMatrix.reset();
				std::ofstream transposeIndexFile("transposeIndexFile.txt") ;
				while ( m_sparseAffineTransformMatrix.next() )
				{
					rowIndex = m_sparseAffineTransformMatrix.getrow();
					colIndex = m_sparseAffineTransformMatrix.getcolumn();
					
					if ( (rowIndex < m_sparseAffineTransformMatrix.rows()) && (colIndex < m_sparseAffineTransformMatrix.cols()) )
						m_sparseTransposeAffineTransformMatrix(colIndex, rowIndex) = m_sparseAffineTransformMatrix.value();
				}

        assert (!outputImageVector.is_zero());
        VectorType outputAffineTransposeImageVector(sizeOfOutputImage[0]*sizeOfOutputImage[1]*sizeOfOutputImage[2]);
        outputAffineTransposeImageVector.fill(0.);

        m_sparseTransposeAffineTransformMatrix.mult(outputImageVector, outputAffineTransposeImageVector);

        std::ofstream vectorAffineTransposeImageFile("vectorAffineTransposeImageFile.txt", std::ios::out | std::ios::app | std::ios::binary);
        vectorAffineTransposeImageFile << outputAffineTransposeImageVector << " ";	

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
    CreateEulerAffineTransformMatrix<IntensityType>
    ::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
        int threadId)
    {
      OutputImageIndexType outIndex;
      OutputImagePointType outPoint;

      ImageRegionIterator<OutputImageType> outputIterator;

      Matrix<double, 4, 4> affineCoreMatrix;


      // Allocate output

      InputImageConstPointer inImage  = this->GetInput();
      OutputImagePointer     outImage = this->GetOutput();

      // Support progress methods/callbacks

      ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

      // Create the affine transformation core matrix

      OutputImageSpacingType res2D = outImage->GetSpacing();

      affineCoreMatrix = this->m_AffineTransform->GetFullAffineMatrix();
      std::cerr << "The core affine transformation matrix is:" << std::endl << m_AffineTransform->GetFullAffineMatrix() << std::endl; 
      m_AffineTransform->InvertTransformationMatrix(); 
      std::cerr << "The inverted core affine transformation matrix is:" << std::endl << m_AffineTransform->GetFullAffineMatrix() << std::endl; 

      // Iterate over pixels in the 2D projection (i.e. output) image

      outputIterator = ImageRegionIterator<OutputImageType>(outImage, outputRegionForThread);

      for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) {

        // Determine the coordinate of the output pixel
        outIndex = outputIterator.GetIndex();
        outImage->TransformIndexToPhysicalPoint(outIndex, outPoint);

      }

    }


  /* -----------------------------------------------------------------------
     AfterThreadedGenerateData()
     ----------------------------------------------------------------------- */

  template< class IntensityType>
    void
    CreateEulerAffineTransformMatrix<IntensityType>
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

      sprintf(filename, "/tmp/CreateEulerAffineTransformMatrix_OUTPUT_%03d.gipl", ++n );
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
