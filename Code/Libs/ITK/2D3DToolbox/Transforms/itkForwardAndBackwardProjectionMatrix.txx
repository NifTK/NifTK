/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkForwardAndBackwardProjectionMatrix_txx
#define __itkForwardAndBackwardProjectionMatrix_txx

#include "itkForwardAndBackwardProjectionMatrix.h"

#include "itkImageRegionIterator.h"
#include "itkProgressReporter.h"

#include "itkCastImageFilter.h"

#include "itkLogHelper.h"


namespace itk
{

  /* -----------------------------------------------------------------------
     Constructor
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    ForwardAndBackwardProjectionMatrix<TScalarType, IntensityType>
    ::ForwardAndBackwardProjectionMatrix()
    {
      m_FlagInitialised = false;

      // Initialise the threshold above which intensities are integrated
      m_Threshold = 0.;

      // Set default values for the output image size

      m_OutputImageSize[0]  = 100;  // size along X
      m_OutputImageSize[1]  = 100;  // size along Y

      // Set default values for the output image resolution

      m_OutputImageSpacing[0]  = 1;  // resolution along X axis
      m_OutputImageSpacing[1]  = 1;  // resolution along Y axis

      // Set default values for the output image origin

      m_OutputImageOrigin[0]  = 0.;  // origin in X
      m_OutputImageOrigin[1]  = 0.;  // origin in Y
    }


  /* -----------------------------------------------------------------------
     PrintSelf(std::ostream&, Indent)
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    ForwardAndBackwardProjectionMatrix<TScalarType, IntensityType>
    ::PrintSelf(std::ostream& os, Indent indent) const
    {
      Superclass::PrintSelf(os,indent);
    }


  /* -----------------------------------------------------------------------
     GetForwardProjectionSparseMatrix()
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    ForwardAndBackwardProjectionMatrix<TScalarType, IntensityType>
    ::GetForwardProjectionSparseMatrix(SparseMatrixType &R, InputImageConstPointer inImage, OutputImagePointer outImage,
        VolumeSizeType &inSize, OutputImageSizeType &outSize, const unsigned int &projNum) 
    {

      // Define a sparse matrix to store the forward projection matrix coefficients
      unsigned long int outSizeTotal = outSize[0]*outSize[1];
      // unsigned long int inSizeTotal  = inSize[0]*inSize[1]*inSize[2];

      // Set the iterator for the projections
      unsigned int iProjection;

      EulerAffineTransformPointer affineTransform;
      PerspectiveProjectionTransformPointer perspTransform;

      for (iProjection=0; iProjection < projNum; iProjection++) {

        niftkitkInfoMacro(<<"Performing forward projection number: " << iProjection);

        double integral = 0;
        OutputImageIndexType outIndex;
        OutputImagePointType outPoint;

        Ray<InputImageType> ray;
        Matrix<double, 4, 4> projMatrix;

        // Create the ray object
        ray.SetImage( inImage );

        perspTransform = m_ProjectionGeometry->GetPerspectiveTransform( iProjection );
        affineTransform = m_ProjectionGeometry->GetAffineTransform( iProjection );

        this->SetPerspectiveTransform( perspTransform );
        this->SetAffineTransform( affineTransform );


        projMatrix = this->m_PerspectiveTransform->GetMatrix();
        projMatrix *= this->m_AffineTransform->GetFullAffineMatrix();

        ray.SetProjectionMatrix(projMatrix);

        // Iterate over pixels in the 2D projection (i.e. output) image
        ImageRegionIterator<OutputImageType> outputIterator;
        outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetLargestPossibleRegion()); 

        // Initialise the index of the intersection point
        double y, z = 0;
        const int* index;

        // Loop over each pixel in the projected 2D image, and from each pixel we cast a ray to the 3D volume
        // Iterate over pixels in the 2D projection (i.e. output) image
        unsigned long int pixel2D = 0;
        unsigned long int xMatrix = 0, yMatrixOne = 0, yMatrixTwo = 0, yMatrixTwoY = 0, yMatrixThree = 0, yMatrixFour = 0, yMatrixFourY = 0;
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

            xMatrix 			= iProjection*outSizeTotal + pixel2D;
            yMatrixOne 		= inSize[2]*inSize[1]*index[2] + inSize[1]*index[1] + index[0];
            yMatrixTwo	 	= inSize[2]*inSize[1]*index[2] + inSize[1]*(index[1]+1) + index[0];
            yMatrixTwoY	 	= inSize[2]*inSize[1]*(index[2]+1) + inSize[1]*index[1] + index[0];
            yMatrixThree 	= inSize[2]*inSize[1]*index[2] + inSize[1]*index[1] + (index[0]+1);
            yMatrixFour 	= inSize[2]*inSize[1]*index[2] + inSize[1]*(index[1]+1) + (index[0]+1);
            yMatrixFourY 	= inSize[2]*inSize[1]*(index[2]+1) + inSize[1]*index[1] + (index[0]+1);

            // std::cerr << "xMatrix value is: " << xMatrix << std::endl;
            // std::cerr << "yMatrixOne value is: " << yMatrixOne << std::endl;

            switch( ray.GetTraversalDirection() )
            {
              case TRANSVERSE_IN_X:
                {
                  R(xMatrix, yMatrixOne) 		= 1. - y*z;
                  R(xMatrix, yMatrixTwo) 		= 1. - z + y*z;
                  R(xMatrix, yMatrixThree) 	= 1. - y + y*z;
                  R(xMatrix, yMatrixFour)		= y + z - y*z;
                  break;
                }
              case TRANSVERSE_IN_Y:
                {
                  R(xMatrix, yMatrixOne) 		= 1. - y*z;
                  R(xMatrix, yMatrixTwoY) 	= 1. - z + y*z;
                  R(xMatrix, yMatrixThree) 	= 1. - y + y*z;
                  R(xMatrix, yMatrixFourY)	= y + z - y*z;
                  break;
                }
              case TRANSVERSE_IN_Z: // Only this case makes the changes
                {
                  R(xMatrix, yMatrixOne) 		= 1. - y*z;
                  R(xMatrix, yMatrixTwo) 		= 1. - z + y*z;
                  R(xMatrix, yMatrixThree) 	= 1. - y + y*z;
                  R(xMatrix, yMatrixFour)		= y + z - y*z;
                  break;
                  /* case TRANSVERSE_IN_Z: // Only this case makes the changes
                     {
                     R(xMatrix, yMatrixOne) 		= (1. - y*z)*normCoef;
                     R(xMatrix, yMatrixTwo) 		= (1. - y + y*z)*normCoef;
                     R(xMatrix, yMatrixThree) 	= (1. - z + y*z)*normCoef;
                     R(xMatrix, yMatrixFour)		= (y + z - y*z)*normCoef;
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

          niftkitkDebugMacro(<<"Finished forward projection: " << iProjection << endl);

        }

        // pSparseForwardProjMatrix = &R;

      }

  /* -----------------------------------------------------------------------
     GetForwardProjectionSparseMatrix()
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    ForwardAndBackwardProjectionMatrix<TScalarType, IntensityType>
    ::GetForwardProjectionSparseMatrix(SparseMatrixType &R, InputImagePointer inImage, OutputImagePointer outImage,
        VolumeSizeType &inSize, OutputImageSizeType &outSize, const unsigned int &projNum) 
    {

			// This function is an overloaded version of GetForwardProjectionSparseMatrix() 
			// using non-const input image pointer.
      // Define a sparse matrix to store the forward projection matrix coefficients
      unsigned long int outSizeTotal = outSize[0]*outSize[1];
      // unsigned long int inSizeTotal  = inSize[0]*inSize[1]*inSize[2];

      // Set the iterator for the projections
      unsigned int iProjection;

      EulerAffineTransformPointer affineTransform;
      PerspectiveProjectionTransformPointer perspTransform;

      for (iProjection=0; iProjection < projNum; iProjection++) {

        niftkitkInfoMacro(<<"Performing forward projection number: " << iProjection);

        double integral = 0;
        OutputImageIndexType outIndex;
        OutputImagePointType outPoint;

        Ray<InputImageType> ray;
        Matrix<double, 4, 4> projMatrix;

        // Create the ray object
        ray.SetImage( inImage );

        perspTransform = m_ProjectionGeometry->GetPerspectiveTransform( iProjection );
        affineTransform = m_ProjectionGeometry->GetAffineTransform( iProjection );

        this->SetPerspectiveTransform( perspTransform );
        this->SetAffineTransform( affineTransform );


        projMatrix = this->m_PerspectiveTransform->GetMatrix();
        projMatrix *= this->m_AffineTransform->GetFullAffineMatrix();

        ray.SetProjectionMatrix(projMatrix);

        // Iterate over pixels in the 2D projection (i.e. output) image
        ImageRegionIterator<OutputImageType> outputIterator;
        outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetLargestPossibleRegion()); 

        // Initialise the index of the intersection point
        double y, z = 0;
        const int* index;

        // Loop over each pixel in the projected 2D image, and from each pixel we cast a ray to the 3D volume
        // Iterate over pixels in the 2D projection (i.e. output) image
        unsigned long int pixel2D = 0;
        unsigned long int xMatrix = 0, yMatrixOne = 0, yMatrixTwo = 0, yMatrixTwoY = 0, yMatrixThree = 0, yMatrixFour = 0, yMatrixFourY = 0;
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

            xMatrix 			= iProjection*outSizeTotal + pixel2D;
            yMatrixOne 		= inSize[2]*inSize[1]*index[2] + inSize[1]*index[1] + index[0];
            yMatrixTwo	 	= inSize[2]*inSize[1]*index[2] + inSize[1]*(index[1]+1) + index[0];
            yMatrixTwoY	 	= inSize[2]*inSize[1]*(index[2]+1) + inSize[1]*index[1] + index[0];
            yMatrixThree 	= inSize[2]*inSize[1]*index[2] + inSize[1]*index[1] + (index[0]+1);
            yMatrixFour 	= inSize[2]*inSize[1]*index[2] + inSize[1]*(index[1]+1) + (index[0]+1);
            yMatrixFourY 	= inSize[2]*inSize[1]*(index[2]+1) + inSize[1]*index[1] + (index[0]+1);

            // std::cerr << "xMatrix value is: " << xMatrix << std::endl;
            // std::cerr << "yMatrixOne value is: " << yMatrixOne << std::endl;

            switch( ray.GetTraversalDirection() )
            {
              case TRANSVERSE_IN_X:
                {
                  R(xMatrix, yMatrixOne) 		= 1. - y*z;
                  R(xMatrix, yMatrixTwo) 		= 1. - z + y*z;
                  R(xMatrix, yMatrixThree) 	= 1. - y + y*z;
                  R(xMatrix, yMatrixFour)		= y + z - y*z;
                  break;
                }
              case TRANSVERSE_IN_Y:
                {
                  R(xMatrix, yMatrixOne) 		= 1. - y*z;
                  R(xMatrix, yMatrixTwoY) 	= 1. - z + y*z;
                  R(xMatrix, yMatrixThree) 	= 1. - y + y*z;
                  R(xMatrix, yMatrixFourY)	= y + z - y*z;
                  break;
                }
              case TRANSVERSE_IN_Z: // Only this case makes the changes
                {
                  R(xMatrix, yMatrixOne) 		= 1. - y*z;
                  R(xMatrix, yMatrixTwo) 		= 1. - z + y*z;
                  R(xMatrix, yMatrixThree) 	= 1. - y + y*z;
                  R(xMatrix, yMatrixFour)		= y + z - y*z;
                  break;
                  /* case TRANSVERSE_IN_Z: // Only this case makes the changes
                     {
                     R(xMatrix, yMatrixOne) 		= (1. - y*z)*normCoef;
                     R(xMatrix, yMatrixTwo) 		= (1. - y + y*z)*normCoef;
                     R(xMatrix, yMatrixThree) 	= (1. - z + y*z)*normCoef;
                     R(xMatrix, yMatrixFour)		= (y + z - y*z)*normCoef;
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

          niftkitkDebugMacro(<<"Finished forward projection: " << iProjection << endl);

        }

        // pSparseForwardProjMatrix = &R;

      }

      /* -----------------------------------------------------------------------
         GetBackwardProjectionSparseMatrix()
         ----------------------------------------------------------------------- */

      template <class TScalarType, class IntensityType>
        void 
        ForwardAndBackwardProjectionMatrix<TScalarType, IntensityType>
        ::GetBackwardProjectionSparseMatrix(SparseMatrixType &R, SparseMatrixType &RTrans, 
            VolumeSizeType &inSize, OutputImageSizeType &outSize, const unsigned int &projNum) 
        {

					// Get the backward projection matrix, which is equivalent to the transpose of the
          // forward projection matrix.
          // Define a sparse matrix to store the affine transformation matrix coefficients
          m_input3DImageTotalSize	 	= inSize[0]*inSize[1]*inSize[2];
          m_output2DImageTotalSize 	= outSize[0]*outSize[1];
          assert ( (RTrans.rows() == m_input3DImageTotalSize) && (RTrans.cols() == m_output2DImageTotalSize*projNum) );

          unsigned long int rowIndex = 0;
          unsigned long int colIndex = 0;
          R.reset();

          while ( R.next() )
          {
            rowIndex = R.getrow();
            colIndex = R.getcolumn();

            if ( (rowIndex < R.rows()) && (colIndex < R.cols()) )
              RTrans(colIndex, rowIndex) = R.value();
          }

          // pSparseBackwardProjMatrix = &RTrans;

        }

      /* -----------------------------------------------------------------------
         CalculteMatrixVectorMultiplication()
         ----------------------------------------------------------------------- */

      template <class TScalarType, class IntensityType>
        void 
        ForwardAndBackwardProjectionMatrix<TScalarType, IntensityType>
        ::CalculteMatrixVectorMultiplication(SparseMatrixType &R, VectorType const& inputImageVector, VectorType &outputImageVector) 
        {

					// This funtion is used to calculate the matrix/vector product
          try { 
            logHelperObject.InfoMessage(std::string("Calculating the multiplication of transformation matrix and image vector."));
            // pSparseAffineTransformMatrix->mult(inputImageVector, outputImageVector);
            R.mult(inputImageVector, outputImageVector);
            logHelperObject.InfoMessage(std::string("Done"));
          } 
          catch( itk::ExceptionObject & err ) { 
            std::cerr << "ERROR: Failed to do the multiplication" << err << std::endl;
          }

        }

    } // end namespace itk


#endif
