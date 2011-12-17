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

#ifndef __itkEulerAffineTransformMatrixAndItsVariations_txx
#define __itkEulerAffineTransformMatrixAndItsVariations_txx

#include "itkEulerAffineTransformMatrixAndItsVariations.h"

#include "itkImageRegionIterator.h"
#include "itkProgressReporter.h"

#include "itkCastImageFilter.h"

#include "itkUCLMacro.h"


namespace itk
{

  /* -----------------------------------------------------------------------
     Constructor
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::EulerAffineTransformMatrixAndItsVariations()
    {
      m_AffineTransform = EulerAffineTransformType::New();
      m_FlagInitialised = false;
    }


  /* -----------------------------------------------------------------------
     PrintSelf(std::ostream&, Indent)
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    void
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::PrintSelf(std::ostream& os, Indent indent) const
    {
      Superclass::PrintSelf(os,indent);
    }


  /* -----------------------------------------------------------------------
     GetAffineTransformationSparseMatrix()
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    void
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::GetAffineTransformationSparseMatrix(SparseMatrixType &R, VolumeSizeType &inSize, EulerAffineTransformParametersType &parameters) 
    {

			// This function works by inputting a blank sparse matrix, the dimension of the volume,
			// and the transformation parameters. It outputs the affine transformation matrix.
      EulerAffineTransformType::InputPointType center;
			center.Fill(0.0);
			// center.Fill(inSize[0]/2);
      // center.SetElement(0, inSize[0]);
			// center.SetElement(1, inSize[1]);
			// center.SetElement(2, inSize[2]);

			// 4x4 affine core matrix and its inverse with homogeneous coordinates
			// In order to get the affine transformation matrix with bilinear interpolation easily,
      // we need the inverse core matrix to calculate the entries of the full affine 
			// transformation matrix. (Google 'inverse mapping')
      Matrix<double, 4, 4> affineCoreMatrix;
      // Matrix<double, 4, 4> affineCoreMatrixInverse;

      affineCoreMatrix.SetIdentity();
      // affineCoreMatrixInverse.SetIdentity();

      VectorType m_inputCoordinateVector(4);
      VectorType m_outputCoordinateVector(4);

      m_AffineTransform->SetCenter(center);
      m_AffineTransform->SetParameters(parameters);

      // Define a sparse matrix to store the affine transformation matrix coefficients
      m_input3DImageTotalSize = inSize[0]*inSize[1]*inSize[2];
      assert ( (R.rows() == m_input3DImageTotalSize) && (R.cols() == m_input3DImageTotalSize) );

      affineCoreMatrix = this->m_AffineTransform->GetFullAffineMatrix();
      m_affineCoreMatrix = affineCoreMatrix.GetVnlMatrix();
      // std::cerr << "The core affine transformation matrix is:" << std::endl << m_AffineTransform->GetFullAffineMatrix() << std::endl;

      // m_AffineTransform->InvertTransformationMatrix();
      // affineCoreMatrixInverse = m_AffineTransform->GetFullAffineMatrix();
      // m_affineCoreMatrixInverse = affineCoreMatrixInverse.GetVnlMatrix();
      // std::cerr << "The inverted core affine transformation matrix is:" << std::endl << m_AffineTransform->GetFullAffineMatrix() << std::endl;

      std::ofstream invTransCoorVectorFile("invTransCoorVectorFile.txt");
      std::ofstream originalCoorVectorFile("originalCoorVectorFile.txt");
      std::ofstream coefFile("coefFile.txt");
      // std::ofstream indexFile("indexFile.txt");
      // indexFile << "The index is :" << std::endl;
      // std::ofstream colIndexNumFile("colIndexNumFile.txt");
      // colIndexNumFile << "The col index is :" << std::endl;
			
			// Get each entries of the full affine transformation matrix with bilinear interpolation
      unsigned long int xCoordin = 0, yCoordin = 0, zCoordin = 0;
      unsigned long int voxelNum = 0;
      unsigned long int colIndexNum1 = 0, colIndexNum2 = 0, colIndexNum3 = 0, colIndexNum4 = 0; 
			unsigned long int colIndexNum5 = 0, colIndexNum6 = 0, colIndexNum7 = 0, colIndexNum8 = 0;
      double xCoorCoef = 0., yCoorCoef = 0., zCoorCoef = 0.;
			double xyCoef = 0., xzCoef = 0., yzCoef = 0., xyzCoef = 0.;
      double leftBottomXCoorinate = 0., leftBottomYCoorinate = 0., leftBottomZCoorinate = 0.;
      unsigned long int intLeftBottomXCoorinate = 0, intLeftBottomYCoorinate = 0, intLeftBottomZCoorinate;
      for ( zCoordin=0; zCoordin < inSize[2]; ++zCoordin )
        for ( yCoordin=0; yCoordin < inSize[1]; ++yCoordin )
          for ( xCoordin=0; xCoordin < inSize[0]; ++xCoordin ) 
          {
            voxelNum = xCoordin + inSize[0]*yCoordin + inSize[1]*inSize[0]*zCoordin;
            // indexFile << voxelNum << std::endl;

            m_outputCoordinateVector.put( 0, xCoordin );
            m_outputCoordinateVector.put( 1, yCoordin );
            m_outputCoordinateVector.put( 2, zCoordin );
            m_outputCoordinateVector.put( 3, 1 );

            originalCoorVectorFile << std::endl << "The original coordinates of voxel number " << voxelNum<< " are:" 
             	<< std::endl << m_outputCoordinateVector[0] << " " << m_outputCoordinateVector[1] << " " 
            	<< m_outputCoordinateVector[2] << " " << m_outputCoordinateVector[3] << std::endl;

            // m_inputCoordinateVector = m_affineCoreMatrixInverse*m_outputCoordinateVector;
						m_inputCoordinateVector = m_affineCoreMatrix*m_outputCoordinateVector;

						// Firstly, we need to exclude the voxels getting out-of-range after the affine transformation
            if ( (m_inputCoordinateVector[0] < 0) || (m_inputCoordinateVector[1] < 0) || (m_inputCoordinateVector[2] < 0) ||
                 (m_inputCoordinateVector[0] > (inSize[0] - 1)) || 
                 (m_inputCoordinateVector[1] > (inSize[1] - 1)) || 
                 (m_inputCoordinateVector[2] > (inSize[2] - 1)) )
            {
              m_inputCoordinateVector[0]=0;
              m_inputCoordinateVector[1]=0;
              m_inputCoordinateVector[2]=0;
            }
            else
            {
              invTransCoorVectorFile << std::endl << "The inverse transformed coordinates of voxel number " << voxelNum<< " are:" 
                << std::endl << m_inputCoordinateVector[0] << " " << m_inputCoordinateVector[1] << " " 
                << m_inputCoordinateVector[2] << " " << m_inputCoordinateVector[3] << std::endl;

              leftBottomXCoorinate = vcl_floor(m_inputCoordinateVector[0]);
              leftBottomYCoorinate = vcl_floor(m_inputCoordinateVector[1]);
              leftBottomZCoorinate = vcl_floor(m_inputCoordinateVector[2]);
              intLeftBottomXCoorinate = (unsigned long int) leftBottomXCoorinate;
              intLeftBottomYCoorinate = (unsigned long int) leftBottomYCoorinate;
              intLeftBottomZCoorinate = (unsigned long int) leftBottomZCoorinate;

              xCoorCoef = (double) vcl_abs(m_inputCoordinateVector[0] - leftBottomXCoorinate);
              yCoorCoef = (double) vcl_abs(m_inputCoordinateVector[1] - leftBottomYCoorinate);
							zCoorCoef = (double) vcl_abs(m_inputCoordinateVector[2] - intLeftBottomZCoorinate);

              coefFile << std::endl << "The coefs are: " << xCoorCoef << " and " << yCoorCoef << " and " << zCoorCoef << std::endl;

              colIndexNum1 		= inSize[1]*inSize[0]*intLeftBottomZCoorinate + inSize[0]*intLeftBottomYCoorinate + intLeftBottomXCoorinate;
              colIndexNum2 		= inSize[1]*inSize[0]*intLeftBottomZCoorinate + inSize[0]*(intLeftBottomYCoorinate+1) + intLeftBottomXCoorinate;
              colIndexNum3 		= inSize[1]*inSize[0]*intLeftBottomZCoorinate + inSize[0]*intLeftBottomYCoorinate + (intLeftBottomXCoorinate+1);
              colIndexNum4	 	= inSize[1]*inSize[0]*intLeftBottomZCoorinate + inSize[0]*(intLeftBottomYCoorinate+1) + (intLeftBottomXCoorinate+1);
              colIndexNum5	 	= inSize[1]*inSize[0]*(intLeftBottomZCoorinate+1) + inSize[0]*intLeftBottomYCoorinate + intLeftBottomXCoorinate;
              colIndexNum6	 	= inSize[1]*inSize[0]*(intLeftBottomZCoorinate+1) + inSize[0]*(intLeftBottomYCoorinate+1) + intLeftBottomXCoorinate;
              colIndexNum7	 	= inSize[1]*inSize[0]*(intLeftBottomZCoorinate+1) + inSize[0]*intLeftBottomYCoorinate + (intLeftBottomXCoorinate+1);
              colIndexNum8	 	= inSize[1]*inSize[0]*(intLeftBottomZCoorinate+1) + inSize[0]*(intLeftBottomYCoorinate+1) + (intLeftBottomXCoorinate+1);

              if ( (colIndexNum1 < m_input3DImageTotalSize)  && (colIndexNum2 < m_input3DImageTotalSize) &&
                   (colIndexNum3 < m_input3DImageTotalSize)  && (colIndexNum4 < m_input3DImageTotalSize) &&
									 (colIndexNum5 < m_input3DImageTotalSize)  && (colIndexNum6 < m_input3DImageTotalSize) &&
									 (colIndexNum7 < m_input3DImageTotalSize)  && (colIndexNum8 < m_input3DImageTotalSize) ) 
              {
                assert( (voxelNum < m_input3DImageTotalSize) && 
												(colIndexNum1 < m_input3DImageTotalSize)  && (colIndexNum2 < m_input3DImageTotalSize) &&
                   			(colIndexNum3 < m_input3DImageTotalSize)  && (colIndexNum4 < m_input3DImageTotalSize) &&
									 			(colIndexNum5 < m_input3DImageTotalSize)  && (colIndexNum6 < m_input3DImageTotalSize) &&
									 			(colIndexNum7 < m_input3DImageTotalSize)  && (colIndexNum8 < m_input3DImageTotalSize) );

								xyCoef = xCoorCoef*yCoorCoef; xzCoef = xCoorCoef*zCoorCoef; yzCoef = yCoorCoef*zCoorCoef; xyzCoef = xCoorCoef*yCoorCoef*zCoorCoef;

								// Secondly, if the affine transformed voxel is overlapped on the left bottom index we have:
								if ( (xCoorCoef == 0.0) && (yCoorCoef == 0.0) && (zCoorCoef == 0.0) )
                	R(voxelNum, colIndexNum1) 		= 1.0;
								// Else, we have:
								else if ( (xCoorCoef == 0.0) && (yCoorCoef == 0.0) && (zCoorCoef != 0.0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - zCoorCoef;
									R(voxelNum, colIndexNum5)  		= zCoorCoef;
								}				
								else if ( (xCoorCoef == 0.0) && (zCoorCoef == 0.0) && (yCoorCoef != 0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - yCoorCoef;
                	R(voxelNum, colIndexNum2) 		= yCoorCoef; 
								}
								else if ( (yCoorCoef == 0.0) && (zCoorCoef == 0.0) && (xCoorCoef != 0.0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - xCoorCoef;
                	R(voxelNum, colIndexNum3)			= xCoorCoef; 
								}
								else if ( (xCoorCoef == 0.0) && (yCoorCoef != 0.0) && (zCoorCoef != 0.0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - yCoorCoef - zCoorCoef + yzCoef;
                	R(voxelNum, colIndexNum5) 		= zCoorCoef - yzCoef; 
                	R(voxelNum, colIndexNum2)			= yCoorCoef - yzCoef; 
                	R(voxelNum, colIndexNum6) 		= yzCoef;
								}
								else if ( (yCoorCoef == 0.0) && (xCoorCoef != 0.0) && (zCoorCoef != 0.0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - xCoorCoef - zCoorCoef + xzCoef;
                	R(voxelNum, colIndexNum5) 		= zCoorCoef - xzCoef; 
                	R(voxelNum, colIndexNum3)			= xCoorCoef - xzCoef;
                	R(voxelNum, colIndexNum7) 		= xzCoef;
								}
								else if ( (zCoorCoef == 0.0) && (xCoorCoef != 0.0) && (yCoorCoef != 0.0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - xCoorCoef - yCoorCoef + xyCoef;
                	R(voxelNum, colIndexNum2) 		= yCoorCoef - xyCoef; 
                	R(voxelNum, colIndexNum3)			= xCoorCoef - xyCoef;
                	R(voxelNum, colIndexNum4) 		= xyCoef;
								}
								else if ( (zCoorCoef != 0.0) && (xCoorCoef != 0.0) && (yCoorCoef != 0.0) )
								{
                	R(voxelNum, colIndexNum1) 		= 1. - xCoorCoef - yCoorCoef - zCoorCoef + xyCoef + xzCoef + yzCoef - xyzCoef;
                	R(voxelNum, colIndexNum2) 		= yCoorCoef - xyCoef - yzCoef + xyzCoef;
                	R(voxelNum, colIndexNum3)			= xCoorCoef - xyCoef - xzCoef  + xyzCoef;
                	R(voxelNum, colIndexNum4) 		= xyCoef - xyzCoef;
                	R(voxelNum, colIndexNum5) 		= zCoorCoef - xzCoef - yzCoef + xyzCoef; 
                	R(voxelNum, colIndexNum6) 		= yzCoef - xyzCoef;
                	R(voxelNum, colIndexNum7) 		= xzCoef - xyzCoef;
                	R(voxelNum, colIndexNum8) 		= xyzCoef;
								}

                // colIndexNumFile << colIndexNumOne << std::endl;		
              }
            } 
          }

      // pSparseAffineTransformMatrix = &R;

    }

  /* -----------------------------------------------------------------------
     GetAffineTransformationSparseMatrixT()
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    void 
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::GetAffineTransformationSparseMatrixT(SparseMatrixType &R, SparseMatrixType &RTrans, 
        VolumeSizeType &inSize) 
    {

			// This function is written to obtain the transpose of the affine transformation matrix.
      // Define a sparse matrix to store the affine transformation matrix coefficients
      m_input3DImageTotalSize = inSize[0]*inSize[1]*inSize[2];
      assert ( (RTrans.rows() == m_input3DImageTotalSize) && (RTrans.cols() == m_input3DImageTotalSize) );

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

      // pSparseTransposeAffineTransformMatrix = &RTrans;

    }

  /* -----------------------------------------------------------------------
     CalculteMatrixVectorMultiplication()
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    void 
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::CalculteMatrixVectorMultiplication(SparseMatrixType &R, VectorType const& inputImageVector, VectorType &outputImageVector) 
    {

			// This funtion is used to calculate the matrix/vector product
      try { 
        niftkitkInfoMacro(<< "Calculating the multiplication of transformation matrix and image vector.");
        // pSparseAffineTransformMatrix->mult(inputImageVector, outputImageVector);
        R.mult(inputImageVector, outputImageVector);
        niftkitkInfoMacro(<< "Done");
      } 
      catch( itk::ExceptionObject & err ) { 
        std::cerr << "ERROR: Failed to do the multiplication" << err << std::endl;
      }	

    }

  /* -----------------------------------------------------------------------
     CalculteMatrixVectorGradient()
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    void 
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::CalculteMatrixVectorGradient(VolumeSizeType &inSize, VectorType const& inputImageVector, 
        VectorType &outputGradVector, EulerAffineTransformParametersType &parameters, unsigned int paraNum) 
    {

	  niftkitkDebugMacro("The FDM difference is: " << m_FDMDiffValue);

      // Get the dimension of the sparse matrix
      m_input3DImageTotalSize = inSize[0]*inSize[1]*inSize[2];

      // Temporary sparse matrix to hold the plus and minus variations in order to use the Finite Difference Method (FDM)
      SparseMatrixType RTempPlus(m_input3DImageTotalSize, m_input3DImageTotalSize);
      SparseMatrixType RTempMinus(m_input3DImageTotalSize, m_input3DImageTotalSize);

      // Change one of the parameters
      EulerAffineTransformParametersType parametersTempPlus 	= parameters;
      EulerAffineTransformParametersType parametersTempMinus 	= parameters;
      parametersTempPlus[paraNum] 	+= m_FDMDiffValue;
      parametersTempMinus[paraNum] 	-= m_FDMDiffValue;

			std::ofstream parametersTempPlusFile("parametersTempPlusFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	parametersTempPlusFile << parametersTempPlus << std::endl;

			std::ofstream parametersTempMinusFile("parametersTempMinusFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	parametersTempMinusFile << parametersTempMinus << std::endl;

      this->GetAffineTransformationSparseMatrix(RTempPlus, inSize, parametersTempPlus);
      this->GetAffineTransformationSparseMatrix(RTempMinus, inSize, parametersTempMinus);

/*
			unsigned long int rowIndex = 0;
  		unsigned long int colIndex = 0;
			
  		RTempPlus.reset();
*/

/*
 			std::ofstream RTempPlusOriginalFullFile("RTempPlusOriginalFullFile.txt");
  		RTempPlusOriginalFullFile << std::endl << "The entries of the affine matrix are: " << std::endl;

 	 		for ( unsigned long int totalEntryIterRow = 0; totalEntryIterRow < RTempPlus.rows(); totalEntryIterRow++ )
				for ( unsigned long int totalEntryIterCol = 0; totalEntryIterCol < RTempPlus.cols(); totalEntryIterCol++ )
  				RTempPlusOriginalFullFile << std::endl << "Row " << totalEntryIterRow << " and column " << totalEntryIterCol 
																		<< " is: " << RTempPlus(totalEntryIterRow, totalEntryIterCol) << std::endl;
*/

/*
  		std::ofstream RTempPlusOriginalFile("RTempPlusOriginal.txt");
  		RTempPlusOriginalFile << std::endl << "The non-zero entries of the affine matrix are: " << std::endl;

  		rowIndex = 0;
  		colIndex = 0;

 	 		while ( RTempPlus.next() )
  		{
  			rowIndex = RTempPlus.getrow();
  			colIndex = RTempPlus.getcolumn();

  			if ( (rowIndex < RTempPlus.rows()) && (colIndex < RTempPlus.cols()) )	
  			RTempPlusOriginalFile << std::endl << "Row " << rowIndex << " and column " << colIndex << " is: " << RTempPlus.value() << std::endl;
  		}
			
			
  		RTempMinus.reset();
  		std::ofstream RTempMinusOriginalFile("RTempMinusOriginal.txt");
  		RTempMinusOriginalFile << std::endl << "The non-zero entries of the affine matrix are: " << std::endl;

  		rowIndex = 0;
  		colIndex = 0;

 	 		while ( RTempMinus.next() )
  		{
  			rowIndex = RTempMinus.getrow();
  			colIndex = RTempMinus.getcolumn();

  			if ( (rowIndex < RTempMinus.rows()) && (colIndex < RTempMinus.cols()) )	
  			RTempMinusOriginalFile << std::endl << "Row " << rowIndex << " and column " << colIndex << " is: " << RTempMinus.value() << std::endl;
  		}
				

      // Perform the FDM
      RTempPlus.subtract(RTempMinus, RTempPlus);

			
  		RTempPlus.reset();
  		std::ofstream RTempPlusFile("RTempPlus.txt");
			RTempPlusFile << std::endl << "The matrix size is " << RTempPlus.rows() << " by " << RTempPlus.cols() << std::endl;
  		RTempPlusFile << std::endl << "The non-zero entries of the affine matrix are: " << std::endl;

  		rowIndex = 0;
  		colIndex = 0;

 	 		while ( RTempPlus.next() )
  		{
  			rowIndex = RTempPlus.getrow();
  			colIndex = RTempPlus.getcolumn();

  			if ( (rowIndex < RTempPlus.rows()) && (colIndex < RTempPlus.cols()) )	
  			RTempPlusFile << std::endl << "Row " << rowIndex << " and column " << colIndex << " is: " << RTempPlus.value() << std::endl;
  		}
*/			

      // RTempPlus 	/= (m_FDMDiffValue);
      // RTempMinus 	/= (m_FDMDiffValue);

      // RTempPlus -= (RTempMinus);

      // Perform the FDM
      SparseMatrixType RTempDifference(m_input3DImageTotalSize, m_input3DImageTotalSize);			
			RTempPlus.subtract(RTempMinus, RTempDifference);

      try { 
        niftkitkInfoMacro(<< "Calculating the gradient.");
        RTempDifference.mult(inputImageVector, outputGradVector);
        outputGradVector /= (2*m_FDMDiffValue);
        niftkitkInfoMacro(<< "Done");

				// std::ofstream RTempPlusTFile("RTempPlusTFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    		// RTempPlusTFile << RTempPlus << std::endl;

				std::ofstream inputImageVectorTFile("inputImageVectorTFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    		inputImageVectorTFile << inputImageVector << std::endl;

				std::ofstream outputGradVectorFile("outputGradVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    		outputGradVectorFile << outputGradVector << std::endl;
      } 
      catch( itk::ExceptionObject & err ) { 
        std::cerr << "ERROR: Failed to do the the gradient Calculation" << err << std::endl;
      }	

    }

  /* -----------------------------------------------------------------------
     Overload CalculteMatrixVectorGradient()
     ----------------------------------------------------------------------- */

  template <class TScalarType>
    void
    EulerAffineTransformMatrixAndItsVariations<TScalarType>
    ::CalculteMatrixVectorGradient(FullMatrixType &jacobianMatrix, VolumeSizeType &inSize, EulerAffineTransformParametersType &parameters) 
    {

			// An overload method to calculate the gradient of the affine parameters using Jacobian
		  EulerAffineTransformType::InputPointType center;
			center.Fill(0.0);
			
			m_AffineTransform->SetCenter(center);
      m_AffineTransform->SetParameters(parameters);

      m_input3DImageTotalSize = inSize[0]*inSize[1]*inSize[2];
      assert ( (jacobianMatrix.rows() == m_input3DImageTotalSize) && (jacobianMatrix.cols() == parameters.Size()) );

			VectorType jacobianVector(parameters.Size());
			VectorType coorVectorTemp(3);
			jacobianVector.fill(0.);
			coorVectorTemp.fill(0.);

			unsigned long int voxelNum = 0;

			typedef Image<double, 3> InputImageType;
			typedef InputImageType::PointType InputImagePointType;

			InputImagePointType inputCoorPoint;
  		inputCoorPoint[0] = 0.;
  		inputCoorPoint[1] = 0.;
  		inputCoorPoint[2] = 0.;
			std::ofstream jacobianArrayFile("jacobianArrayFile.txt", std::ios::out | std::ios::app | std::ios::binary);
			std::ofstream jacobianMatrixFile("jacobianMatrixFile.txt", std::ios::out | std::ios::app | std::ios::binary);
      for ( unsigned long int zCoordin = 0; zCoordin < inSize[2]; ++zCoordin )
        for ( unsigned long int yCoordin = 0; yCoordin < inSize[1]; ++yCoordin )
          for ( unsigned long int xCoordin = 0; xCoordin < inSize[0]; ++xCoordin ) 
          {

						for ( unsigned long int iCoor = 0; iCoor < 3; iCoor++ )
							coorVectorTemp[iCoor] = inputCoorPoint[iCoor];

						inputCoorPoint[0] = xCoordin;
  					inputCoorPoint[1] = yCoordin;
  					inputCoorPoint[2] = zCoordin;

						m_JacobianArray = m_AffineTransform->GetJacobian(inputCoorPoint);
						
						jacobianVector = coorVectorTemp.pre_multiply(m_JacobianArray);

						voxelNum = xCoordin + inSize[0]*yCoordin + inSize[1]*inSize[0]*zCoordin;
						jacobianMatrix.set_row(voxelNum, jacobianVector);
			
    				jacobianArrayFile << m_JacobianArray << std::endl;
					}

    			jacobianMatrixFile << jacobianMatrix << std::endl;

#if 0
			// An overload method to calculate the gradient of the affine parameters using Jacobian
		  EulerAffineTransformType::InputPointType center;
			center.Fill(0.0);
			// center.Fill(inSize[0]/2);
      // center.SetElement(0, inSize[0]);
			// center.SetElement(1, inSize[1]);
			// center.SetElement(2, inSize[2]);

      m_AffineTransform->SetCenter(center);
      m_AffineTransform->SetParameters(parameters);

      // Define a sparse matrix to store the affine transformation matrix coefficients
      m_input3DImageTotalSize = inSize[0]*inSize[1]*inSize[2];
      assert ( (jacobianMatrix.rows() == m_input3DImageTotalSize) && (jacobianMatrix.cols() == parameters.Size()) );

			// Get the Jacobian of the affine transformation
			unsigned long int voxelNum = 0;

			VectorType jacobianVector(parameters.Size());
			VectorType coorVectorTemp(3);
			jacobianVector.fill(0.);
			coorVectorTemp.fill(0.);

			FullMatrixType jacobianMatrixCore(3, parameters.Size(), 0.);

			typedef Image<float, 3> InputImageType;
			typedef InputImageType::PointType InputImagePointType;

			InputImagePointType inputCoorPoint;
  		inputCoorPoint[0] = 0.;
  		inputCoorPoint[1] = 0.;
  		inputCoorPoint[2] = 0.;
      for ( unsigned long int zCoordin = 0; zCoordin < inSize[2]; ++zCoordin )
        for ( unsigned long int yCoordin = 0; yCoordin < inSize[1]; ++yCoordin )
          for ( unsigned long int xCoordin = 0; xCoordin < inSize[0]; ++xCoordin ) 
          {
						
						voxelNum = xCoordin + inSize[0]*yCoordin + inSize[1]*inSize[0]*zCoordin;

						inputCoorPoint[0] = xCoordin;
  					inputCoorPoint[1] = yCoordin;
  					inputCoorPoint[2] = zCoordin;

						for ( unsigned long int iCoor = 0; iCoor < 3; iCoor++ )
							coorVectorTemp[iCoor] = inputCoorPoint[iCoor];

						// VectorType coorVectorTemp(parameters.Size(), parameters.Size(), inputCoorPoint);    
       
						m_JacobianArray = m_AffineTransform->GetJacobian(inputCoorPoint); 

						for ( unsigned long int iJacobianX = 0; iJacobianX < 3; iJacobianX++ )
							for ( unsigned long int iJacobianY = 0; iJacobianY < parameters.Size(); iJacobianY++ )
								jacobianMatrixCore[iJacobianX][iJacobianY] = m_JacobianArray[iJacobianX][iJacobianY];					

						jacobianVector = coorVectorTemp.post_multiply(jacobianMatrix);
						jacobianMatrix.set_row(voxelNum, jacobianVector);

						// std::ofstream jacobianVectorFile("jacobianVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    				// jacobianVectorFile << jacobianVector << std::endl;

          }
#endif

		}


} // end namespace itk


#endif
