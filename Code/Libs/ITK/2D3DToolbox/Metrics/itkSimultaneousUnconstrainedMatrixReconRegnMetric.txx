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

#ifndef __itkSimultaneousUnconstrainedMatrixReconRegnMetric_txx
#define __itkSimultaneousUnconstrainedMatrixReconRegnMetric_txx

#include "itkSimultaneousUnconstrainedMatrixReconRegnMetric.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"


namespace itk
{

  /* -----------------------------------------------------------------------
     Constructor
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::SimultaneousUnconstrainedMatrixReconRegnMetric()
    {

    }


  /* -----------------------------------------------------------------------
     PrintSelf
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::PrintSelf(std::ostream& os, Indent indent) const
    {
      Superclass::PrintSelf(os,indent);
      os << indent << "CreateForwardBackwardProjectionMatrix: " << std::endl;
    }


  /* -----------------------------------------------------------------------
     GetNumberOfParameters()
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    unsigned int 
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::GetNumberOfParameters( void ) const
    {

      assert ( (m_EstimatedVolumeVector.size()!=0) && (m_TransformationParameterVector.size()!=0) );

      unsigned int nParameters = m_EstimatedVolumeVector.size() + m_TransformationParameterVector.size();
      return nParameters;

    }


  /* -----------------------------------------------------------------------
     Initialise()
     ----------------------------------------------------------------------- */

	template <class TScalarType, class IntensityType>
		void
		SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
		::Initialise(void)
		{

      // Allocate the reconstruction estimate volume
			InputVolumePointer m_inVolume;
      if (m_inVolume.IsNull()) {

    	niftkitkDebugMacro(<< "Allocating the initial volume estimate");

        m_inVolume = InputVolumeType::New();

        InputVolumeRegionType region;
        region.SetSize( m_InVolumeSize );

        m_inVolume->SetRegions( region );
        m_inVolume->SetSpacing( m_InVolumeSpacing );
        m_inVolume->SetOrigin(  m_InVolumeOrigin );

        m_inVolume->Allocate();
        m_inVolume->FillBuffer( 0.1 );
      }

/*
			// Initialise the transformation parameters
			// Should we initialise the 12 affine parameters within Metric's Initialise() function?
			double iniGuessTranPara[12] = { -2.2, 2.1, 3.1, 0, 11, 9, 1, 1, 1, 0, 0, 0 };
			pEulerAffineParameters = iniGuessTranPara;

			std::ofstream pEulerAffineParametersFile("pEulerAffineParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
			for (unsigned int i; i < 12; i++)
    		pEulerAffineParametersFile << pEulerAffineParameters[i] << " ";
				pEulerAffineParametersFile << std::endl;
*/

		}


  /* -----------------------------------------------------------------------
     GetValue() - Get the value of the similarity metric
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    typename SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>::MeasureType
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::GetValue( const ParametersType &parameters ) const
    {

			std::cerr << "The size of the parameters is: " << parameters.size() << std::endl;

			std::ofstream MTransformationParameterVectorFile("MTransformationParameterVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	MTransformationParameterVectorFile << m_TransformationParameterVector << " " << std::endl;

			// Extract the estimate input volume and the transformation parameters
			VectorType m_EstimatedVolumeVector = parameters.extract(m_totalSize3D, 0);
      VectorType m_TransformationParameterVector = parameters.extract(m_paraNumber, m_totalSize3D);

			std::ofstream M2TransformationParameterVectorFile("M2TransformationParameterVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	M2TransformationParameterVectorFile << m_TransformationParameterVector << " " << std::endl;

			// for (unsigned int i = 0; i < m_paraNumber; i++)
			// 	std::cerr << "Transformation parameters: " << m_TransformationParameterVector[i] << " ";

/*
      // Allocate the reconstruction estimate volume
			InputVolumePointer m_inVolume;
      if (m_inVolume.IsNull()) {

        niftkitkDebugMacro(<< "Allocating the initial volume estimate");

        m_inVolume = InputVolumeType::New();

        InputVolumeRegionType region;
        region.SetSize( m_InVolumeSize );

        m_inVolume->SetRegions( region );
        m_inVolume->SetSpacing( m_InVolumeSpacing );
        m_inVolume->SetOrigin(  m_InVolumeOrigin );

        m_inVolume->Allocate();
        m_inVolume->FillBuffer( 0.1 );
      }
*/

      // Change the updated input volume vector into 3D image
      InputVolumeIndexType inIndex;

      ImageRegionIterator<InputVolumeType> inVolumeIterator;
      inVolumeIterator = ImageRegionIterator<InputVolumeType>(m_inVolume, m_inVolume->GetLargestPossibleRegion());

      unsigned long int voxelNumber = 0;
      for ( inVolumeIterator.GoToBegin(); !inVolumeIterator.IsAtEnd(); ++inVolumeIterator)
      {

        // Determine the coordinate of the input volume
        inIndex = inVolumeIterator.GetIndex();
        m_inVolume->SetPixel(inIndex, m_EstimatedVolumeVector[voxelNumber]);

        voxelNumber++;

      }

			// Allocate the matrix projector
			MatrixProjectorPointerType	m_MatrixProjector;
      if ( m_MatrixProjector.IsNull() )
        m_MatrixProjector = MatrixProjectorType::New();

			InputVolumeSizeType inVolumeSize 		= m_InVolumeSize;
			InputProjectionSizeType inProjSize 	= m_InProjectionSize;

			// Create the corresponding forward/backward projection matrix
      const unsigned long int totalSizeAllProjs = m_ProjectionNumber*m_totalSize2D;
      static SparseMatrixType forwardProjectionMatrix(totalSizeAllProjs, m_totalSize3D);
      static SparseMatrixType backwardProjectionMatrix(m_totalSize3D, totalSizeAllProjs);

			// Set the projection geometry
			m_MatrixProjector->SetProjectionGeometry( m_Geometry );

      m_MatrixProjector->GetForwardProjectionSparseMatrix(forwardProjectionMatrix, m_inVolume, m_inProjTemp, 
           inVolumeSize, inProjSize, m_ProjectionNumber);


      // Allocate the affine transformer
			AffineTransformerType::Pointer m_AffineTransformer;
			if ( m_AffineTransformer.IsNull() )
				m_AffineTransformer = AffineTransformerType::New();


/*
			// Modify the transformation parameters
			// EulerAffineTransformType::ParametersType m_EulerAffineParameters(m_paraNumber);
			EulerAffineTransformType::ParametersType pEulerAffineParameters(m_paraNumber);

			double* pIniGuessTranPara;
			// m_EulerAffineParameters.Fill(0);
			pEulerAffineParameters->Fill(0);

			double iniGuessTranPara[12] = { -2.2, 2.1, 3.2, 0.01, 11, 6, 1.1, 1.1, 1.1, 0.01, 0.01, 0.01 };
			if ( !pIniGuessTranPara )
			{
				pIniGuessTranPara = iniGuessTranPara;
				// m_EulerAffineParameters.SetData(iniGuessTranPara, m_paraNumber);
				pEulerAffineParameters->SetData(iniGuessTranPara, m_paraNumber);
			}
			else
				// m_EulerAffineParameters = m_TransformationParameterVector;
			 	pEulerAffineParameters = m_TransformationParameterVector;	

			unsigned int iPara;
			for (iPara = 0; iPara < m_paraNumber; iPara++)
				// // m_EulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				// m_EulerAffineParameters.SetElement(iPara, m_TransformationParameterVector[iPara]);
		 		// pEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				pEulerAffineParameters->SetElement(iPara, m_TransformationParameterVector[iPara]);

*/

			EulerAffineTransformType::ParametersType tempEulerAffineParameters(m_paraNumber);
			for (unsigned int iPara = 0; iPara < m_paraNumber; iPara++)
				// tempEulerAffineParameters.SetElement(iPara, m_TransformationParameterVector[iPara]);
		 		tempEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];

			std::ofstream tempEulerAffineParametersFile("tempEulerAffineParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	tempEulerAffineParametersFile << tempEulerAffineParameters << " ";

/*
			// Modify the transformation parameters			
			EulerAffineTransformType::ParametersType tempEulerAffineParameters(m_paraNumber);
			for (unsigned int iPara = 0; iPara < m_paraNumber; iPara++)
			{
				pEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				tempEulerAffineParameters[iPara] = pEulerAffineParameters[iPara];
			}

			std::ofstream tempEulerAffineParametersFile("tempEulerAffineParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	tempEulerAffineParametersFile << tempEulerAffineParameters << " ";
*/

			// Create the corresponding transformation matrix
 	 		static SparseMatrixType affineMatrix(m_totalSize3D, m_totalSize3D);
  		static SparseMatrixType affineMatrixTranspose(m_totalSize3D, m_totalSize3D);

  		m_AffineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, inVolumeSize, tempEulerAffineParameters);


/*
		  // Print out the all the entries of the sparse affine matrix
			affineMatrix.reset();
  		std::ofstream sparseAffineFullMatrixFile("sparseAffineFullMatrix.txt");

 			for ( unsigned long int totalEntryIterRow = 0; totalEntryIterRow < affineMatrix.rows(); totalEntryIterRow++ )
				for ( unsigned long int totalEntryIterCol = 0; totalEntryIterCol < affineMatrix.cols(); totalEntryIterCol++ )
  				sparseAffineFullMatrixFile << affineMatrix(totalEntryIterRow, totalEntryIterCol) << " ";
*/


  		// Calculate the matrix/vector multiplication in order to get the forward projection (Ax)
  		// assert (!m_EstimatedVolumeVector.is_zero());
  		VectorType forwardProjectedVectorOne(m_totalSize3D);
  		forwardProjectedVectorOne.fill(0.);

  		m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVectorOne);


  		// Calculate the matrix/vector multiplication in order to get the affine transformation (Rx)
  		VectorType affineTransformedVector(m_totalSize3D);
  		affineTransformedVector.fill(0.);

		  m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, m_EstimatedVolumeVector, affineTransformedVector);

  		// Calculate the matrix/vector multiplication in order to get the forward projection (ARx)
  		// assert (!affineTransformedVector.is_zero());
  	  VectorType forwardProjectedVectorTwo(m_totalSize3D);
  		forwardProjectedVectorTwo.fill(0.);
			
			m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, affineTransformedVector, forwardProjectedVectorTwo);

			
			// Initialise the current measure
      MeasureType currentMeasure;
      currentMeasure = 0.;

			// Calculate (Ax - y_1) and (ARx - y_2)
			VectorType	m_inProjOneSub(m_totalSize3D);
			VectorType	m_inProjTwoSub(m_totalSize3D);
			m_inProjOneSub.fill(0.);
			m_inProjTwoSub.fill(0.);
			
			assert( !(this->m_inProjOne.is_zero()) && !(this->m_inProjTwo.is_zero()) );
			m_inProjOneSub = forwardProjectedVectorOne - this->m_inProjOne;
			m_inProjTwoSub = forwardProjectedVectorTwo - this->m_inProjTwo;
			
			std::ofstream projOneVectorFile("projOneVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	projOneVectorFile << m_inProjOneSub << " ";

			std::ofstream projTwoVectorFile("projTwoVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	projTwoVectorFile << m_inProjTwoSub << " ";

			std::ofstream estimateVolumeFile("estimateVolumeFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	estimateVolumeFile << m_EstimatedVolumeVector << " ";

			
			// Calculate (||Ax - y_1||^2 + ||ARx - y_2||^2), which is the cost function value
			currentMeasure = m_inProjOneSub.squared_magnitude() + m_inProjTwoSub.squared_magnitude();
			std::cerr << "Current cost function value is: " << currentMeasure << std::endl;

      return currentMeasure;

    }


  /* -----------------------------------------------------------------------
     GetDerivative() - Get the derivative of the similarity metric
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::GetDerivative( const ParametersType &parameters, 
        DerivativeType &derivative ) const
    {

			std::cerr << "The size of the parameters is: " << parameters.size() << std::endl;
			derivative.set_size( parameters.size() );
			std::cerr << "The size of the derivative is: " << derivative.size() << std::endl;

			// Extract the estimate input volume and the transformation parameters
			VectorType m_EstimatedVolumeVector = parameters.extract(m_totalSize3D, 0);
      VectorType m_TransformationParameterVector = parameters.extract(m_paraNumber, m_totalSize3D);

/*
      // Allocate the reconstruction estimate volume
			InputVolumePointer m_inVolume;
      if (m_inVolume.IsNull()) {

        niftkitkDebugMacro(<< "Allocating the initial volume estimate");

        m_inVolume = InputVolumeType::New();

        InputVolumeRegionType region;
        region.SetSize( m_InVolumeSize );

        m_inVolume->SetRegions( region );
        m_inVolume->SetSpacing( m_InVolumeSpacing );
        m_inVolume->SetOrigin(  m_InVolumeOrigin );

        m_inVolume->Allocate();
        m_inVolume->FillBuffer( 0.1 );
      }
*/

      // Change the updated input volume vector into 3D image
      InputVolumeIndexType inIndex;

      ImageRegionIterator<InputVolumeType> inVolumeIterator;
      inVolumeIterator = ImageRegionIterator<InputVolumeType>(m_inVolume, m_inVolume->GetLargestPossibleRegion());

      unsigned long int voxelNumber = 0;
      for ( inVolumeIterator.GoToBegin(); !inVolumeIterator.IsAtEnd(); ++inVolumeIterator)
      {

        // Determine the coordinate of the input volume
        inIndex = inVolumeIterator.GetIndex();
        m_inVolume->SetPixel(inIndex, m_EstimatedVolumeVector[voxelNumber]);

        voxelNumber++;

      }

			// Allocate the matrix projector
			MatrixProjectorPointerType	m_MatrixProjector;
      if ( m_MatrixProjector.IsNull() )
        m_MatrixProjector = MatrixProjectorType::New();

			InputVolumeSizeType inVolumeSize 		= m_InVolumeSize;
			InputProjectionSizeType inProjSize 	= m_InProjectionSize;

			// Create the corresponding forward/backward projection matrix
      const unsigned long int totalSizeAllProjs = m_ProjectionNumber*m_totalSize2D;
      static SparseMatrixType forwardProjectionMatrix(totalSizeAllProjs, m_totalSize3D);
      static SparseMatrixType backwardProjectionMatrix(m_totalSize3D, totalSizeAllProjs);

			// Set the projection geometry
			m_MatrixProjector->SetProjectionGeometry( m_Geometry );

      m_MatrixProjector->GetForwardProjectionSparseMatrix(forwardProjectionMatrix, m_inVolume, m_inProjTemp, 
           inVolumeSize, inProjSize, m_ProjectionNumber);
      m_MatrixProjector->GetBackwardProjectionSparseMatrix(forwardProjectionMatrix, backwardProjectionMatrix, 
         inVolumeSize, inProjSize, m_ProjectionNumber);


      // Allocate the affine transformer
			AffineTransformerType::Pointer m_AffineTransformer;
			if ( m_AffineTransformer.IsNull() )
				m_AffineTransformer = AffineTransformerType::New();

/*
			// Modify the transformation parameters
			// EulerAffineTransformType::ParametersType m_EulerAffineParameters(m_paraNumber);
			EulerAffineTransformType::ParametersType pEulerAffineParameters(m_paraNumber);

			double* pIniGuessTranPara;
			// m_EulerAffineParameters.Fill(0);
			pEulerAffineParameters->Fill(0);

			double iniGuessTranPara[12] = { -2.2, 2.1, 3.2, 0.01, 11, 6, 1.1, 1.1, 1.1, 0.01, 0.01, 0.01 };
			if ( !pIniGuessTranPara )
			{
				pIniGuessTranPara = iniGuessTranPara;
				// m_EulerAffineParameters.SetData(iniGuessTranPara, m_paraNumber);
				pEulerAffineParameters->SetData(iniGuessTranPara, m_paraNumber);
			}
			else
				// m_EulerAffineParameters = m_TransformationParameterVector;
			 	pEulerAffineParameters = m_TransformationParameterVector;	
	
			unsigned int iPara;
			for (iPara = 0; iPara < m_paraNumber; iPara++)
				m_EulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				// m_EulerAffineParameters.SetElement(iPara, m_TransformationParameterVector[iPara]);
		 		// // pEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				// pEulerAffineParameters->SetElement(iPara, m_TransformationParameterVector[iPara]);
*/

			EulerAffineTransformType::ParametersType tempEulerAffineParameters(m_paraNumber);
			for (unsigned int iPara = 0; iPara < m_paraNumber; iPara++)
				// tempEulerAffineParameters.SetElement(iPara, m_TransformationParameterVector[iPara]);
		 		tempEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];

			std::ofstream tempEulerAffineParametersFile("tempEulerAffineParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	tempEulerAffineParametersFile << tempEulerAffineParameters << " ";

/*
			// Modify the transformation parameters			
			EulerAffineTransformType::ParametersType tempEulerAffineParameters(m_paraNumber);
			for (unsigned int iPara = 0; iPara < m_paraNumber; iPara++)
			{
				pEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				tempEulerAffineParameters[iPara] = pEulerAffineParameters[iPara];
			}

			std::ofstream tempEulerAffineParametersFile("tempEulerAffineParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	tempEulerAffineParametersFile << tempEulerAffineParameters << " ";
*/

			std::ofstream TransformationParameterVectorFile("TransformationParameterVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	TransformationParameterVectorFile << m_TransformationParameterVector << " " << std::endl;

			// Create the corresponding transformation matrix
 	 		static SparseMatrixType affineMatrix(m_totalSize3D, m_totalSize3D);
  		static SparseMatrixType affineMatrixTranspose(m_totalSize3D, m_totalSize3D);

  		m_AffineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, inVolumeSize, tempEulerAffineParameters);
  		m_AffineTransformer->GetAffineTransformationSparseMatrixT(affineMatrix, affineMatrixTranspose, inVolumeSize);

		  // Print out the all the entries of the sparse affine matrix
			affineMatrix.reset();
  		std::ofstream sparseAffineFullMatrixInGetDerivativeFile("sparseAffineFullMatrixInGetDerivative.txt");


/*
 			for ( unsigned long int totalEntryIterRow = 0; totalEntryIterRow < affineMatrix.rows(); totalEntryIterRow++ )
				for ( unsigned long int totalEntryIterCol = 0; totalEntryIterCol < affineMatrix.cols(); totalEntryIterCol++ )
  				sparseAffineFullMatrixInGetDerivativeFile << affineMatrix(totalEntryIterRow, totalEntryIterCol) << " ";

			affineMatrixTranspose.reset();
  		std::ofstream sparseAffineFullMatrixInGetDerivativeTransposeFile("sparseAffineFullMatrixInGetDerivativeTranspose.txt");

 			for ( unsigned long int totalEntryIterRow = 0; totalEntryIterRow < affineMatrixTranspose.rows(); totalEntryIterRow++ )
				for ( unsigned long int totalEntryIterCol = 0; totalEntryIterCol < affineMatrixTranspose.cols(); totalEntryIterCol++ )
  				sparseAffineFullMatrixInGetDerivativeTransposeFile << affineMatrixTranspose(totalEntryIterRow, totalEntryIterCol) << " ";
*/


  		// Calculate the matrix/vector multiplication in order to get the forward projection (Ax)
  		// assert (!m_EstimatedVolumeVector.is_zero());
  		VectorType forwardProjectedVectorOne(m_totalSize3D);
  		forwardProjectedVectorOne.fill(0.);

  		m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVectorOne);


  		// Calculate the matrix/vector multiplication in order to get the affine transformation (Rx)
  		VectorType affineTransformedVector(m_totalSize3D);
  		affineTransformedVector.fill(0.);

		  m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, m_EstimatedVolumeVector, affineTransformedVector);

  		// Calculate the matrix/vector multiplication in order to get the forward projection (ARx)
  		// assert (!affineTransformedVector.is_zero());
  		VectorType forwardProjectedVectorTwo(m_totalSize3D);
  		forwardProjectedVectorTwo.fill(0.);
			
			m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, affineTransformedVector, forwardProjectedVectorTwo);


			// Calculate (Ax - y_1) and (ARx - y_2)
			VectorType	m_inProjOneSub(m_totalSize3D);
			VectorType	m_inProjTwoSub(m_totalSize3D);
			m_inProjOneSub.fill(0.);
			m_inProjTwoSub.fill(0.);
			
			m_inProjOneSub = forwardProjectedVectorOne - this->m_inProjOne;
			m_inProjTwoSub = forwardProjectedVectorTwo - this->m_inProjTwo;

			
			// Process the backprojection (A^T (Ax - y_1)) and (A^T (ARx - y_2))
			// assert (!m_inProjOne.is_zero() && !m_inProjTwo.is_zero());
			VectorType	inBackProjOne(m_totalSize3D); 
			VectorType	inBackProjTwo(m_totalSize3D);
			inBackProjOne.fill(0.);
			inBackProjTwo.fill(0.);

			m_MatrixProjector->CalculteMatrixVectorMultiplication(backwardProjectionMatrix, m_inProjOneSub, inBackProjOne);
			m_MatrixProjector->CalculteMatrixVectorMultiplication(backwardProjectionMatrix, m_inProjTwoSub, inBackProjTwo);

			// Obtain the transpose of affine transformation matrix with the backprojection set two (R^T A^T (ARx - y_2))
			// assert (!inBackProjOne.is_zero() && !inBackProjTwo.is_zero());
			VectorType	inAffineTransposeBackProjTwo(m_totalSize3D);
			inAffineTransposeBackProjTwo.fill(0.);
			
			m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrixTranspose, inBackProjTwo, inAffineTransposeBackProjTwo);


			// Update the derivative with respect to voxel values x by using (A^T (Ax - y_1) + R^T A^T (ARx - y_2))
			derivative.update((inBackProjOne + inAffineTransposeBackProjTwo), 0);


			// Get the derivative with respect to the transformation parameters which is the cross product as ((A^T (ARx - y_2))^T (R'x))
			VectorType	outputVectorGradTemp(m_totalSize3D);
			VectorType  derivativeParameters(m_paraNumber);
			outputVectorGradTemp.fill(0.);
			derivativeParameters.fill(0.);
			
			// Calculate the derivative for each parameter
			assert( !inBackProjTwo.is_zero() );
			for ( unsigned int iDerivative = 0; iDerivative < m_paraNumber; iDerivative++ )
			{

				m_AffineTransformer->CalculteMatrixVectorGradient(inVolumeSize, m_EstimatedVolumeVector, outputVectorGradTemp, tempEulerAffineParameters, iDerivative);
				derivativeParameters[iDerivative] = dot_product( inBackProjTwo, outputVectorGradTemp );

			}

			std::cerr << "Backprojection size: "  << inBackProjTwo.size() 				<< std::endl;
			std::cerr << "Gradient vector size: " << outputVectorGradTemp.size() 	<< std::endl;

			std::ofstream inBackProjTwoFile("inBackProjTwoFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	inBackProjTwoFile << inBackProjTwo << std::endl;

			std::ofstream outputVectorGradTempFile("outputVectorGradTempFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	outputVectorGradTempFile << outputVectorGradTemp << std::endl;

			std::ofstream derivativeParametersFile("derivativeParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	derivativeParametersFile << derivativeParameters << std::endl;

			// Update the derivative with respect to transformation parameters by using ((A^T (ARx - y_2))^T (R'x))
			derivative.update(derivativeParameters, m_totalSize3D);

			std::ofstream derivativeVectorFile("derivativeVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	derivativeVectorFile << derivative << std::endl;
			

    }


  /* -----------------------------------------------------------------------
     GetDerivative() - Get the derivative of the similarity metric
     ----------------------------------------------------------------------- */

/*
  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::GetDerivative( const ParametersType &parameters, 
        DerivativeType &derivative ) const
    {

			std::cerr << "The size of the parameters is: " << parameters.size() << std::endl;
			derivative.set_size( parameters.size() );
			std::cerr << "The size of the derivative is: " << derivative.size() << std::endl;

			// Extract the estimate input volume and the transformation parameters
			VectorType m_EstimatedVolumeVector = parameters.extract(m_totalSize3D, 0);
      VectorType m_TransformationParameterVector = parameters.extract(m_paraNumber, m_totalSize3D);


      // Allocate the reconstruction estimate volume
			InputVolumePointer m_inVolume;
      if (m_inVolume.IsNull()) {

        niftkitkDebugMacro(<< "Allocating the initial volume estimate");

        m_inVolume = InputVolumeType::New();

        InputVolumeRegionType region;
        region.SetSize( m_InVolumeSize );

        m_inVolume->SetRegions( region );
        m_inVolume->SetSpacing( m_InVolumeSpacing );
        m_inVolume->SetOrigin(  m_InVolumeOrigin );

        m_inVolume->Allocate();
        m_inVolume->FillBuffer( 0.1 );
      }

      // Change the updated input volume vector into 3D image
      InputVolumeIndexType inIndex;

      ImageRegionIterator<InputVolumeType> inVolumeIterator;
      inVolumeIterator = ImageRegionIterator<InputVolumeType>(m_inVolume, m_inVolume->GetLargestPossibleRegion());

      unsigned long int voxelNumber = 0;
      for ( inVolumeIterator.GoToBegin(); !inVolumeIterator.IsAtEnd(); ++inVolumeIterator)
      {

        // Determine the coordinate of the input volume
        inIndex = inVolumeIterator.GetIndex();
        m_inVolume->SetPixel(inIndex, m_EstimatedVolumeVector[voxelNumber]);

        voxelNumber++;

      }

			// Allocate the matrix projector
			MatrixProjectorPointerType	m_MatrixProjector;
      if ( m_MatrixProjector.IsNull() )
        m_MatrixProjector = MatrixProjectorType::New();

			InputVolumeSizeType inVolumeSize 		= m_InVolumeSize;
			InputProjectionSizeType inProjSize 	= m_InProjectionSize;

			// Create the corresponding forward/backward projection matrix
      const unsigned long int totalSizeAllProjs = m_ProjectionNumber*m_totalSize2D;
      static SparseMatrixType forwardProjectionMatrix(totalSizeAllProjs, m_totalSize3D);
      static SparseMatrixType backwardProjectionMatrix(m_totalSize3D, totalSizeAllProjs);

			// Set the projection geometry
			m_MatrixProjector->SetProjectionGeometry( m_Geometry );

      m_MatrixProjector->GetForwardProjectionSparseMatrix(forwardProjectionMatrix, m_inVolume, m_inProjTemp, 
           inVolumeSize, inProjSize, m_ProjectionNumber);
      m_MatrixProjector->GetBackwardProjectionSparseMatrix(forwardProjectionMatrix, backwardProjectionMatrix, 
         inVolumeSize, inProjSize, m_ProjectionNumber);


      // Allocate the affine transformer
			AffineTransformerType::Pointer m_AffineTransformer;
			if ( m_AffineTransformer.IsNull() )
				m_AffineTransformer = AffineTransformerType::New();

			// Modify the transformation parameters
			EulerAffineTransformType::ParametersType pEulerAffineParameters(m_paraNumber);
			unsigned int iPara;
			for (iPara = 0; iPara < m_paraNumber; iPara++)
				pEulerAffineParameters[iPara] = m_TransformationParameterVector[iPara];
				// pEulerAffineParameters->SetElement(iPara, m_TransformationParameterVector[iPara]);

			std::ofstream EulerAffineParametersFile("EulerAffineParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	EulerAffineParametersFile << pEulerAffineParameters << " ";

			std::ofstream TransformationParameterVectorFile("TransformationParameterVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	TransformationParameterVectorFile << m_TransformationParameterVector << " " << std::endl;

			// Create the corresponding transformation matrix
 	 		static SparseMatrixType affineMatrix(m_totalSize3D, m_totalSize3D);
  		static SparseMatrixType affineMatrixTranspose(m_totalSize3D, m_totalSize3D);

  		m_AffineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, inVolumeSize, pEulerAffineParameters);
  		m_AffineTransformer->GetAffineTransformationSparseMatrixT(affineMatrix, affineMatrixTranspose, inVolumeSize);


  		// Calculate the matrix/vector multiplication in order to get the forward projection (Ax)
  		// assert (!m_EstimatedVolumeVector.is_zero());
  		VectorType forwardProjectedVectorOne(m_totalSize3D);
  		forwardProjectedVectorOne.fill(0.);

  		m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVectorOne);


  		// Calculate the matrix/vector multiplication in order to get the affine transformation (Rx)
  		VectorType affineTransformedVector(m_totalSize3D);
  		affineTransformedVector.fill(0.);

		  m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, m_EstimatedVolumeVector, affineTransformedVector);

  		// Calculate the matrix/vector multiplication in order to get the forward projection (ARx)
  		// assert (!affineTransformedVector.is_zero());
  		VectorType forwardProjectedVectorTwo(m_totalSize3D);
  		forwardProjectedVectorTwo.fill(0.);
			
			m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, affineTransformedVector, forwardProjectedVectorTwo);


			// Calculate (Ax - y_1) and (ARx - y_2)
			VectorType	m_inProjOneSub(m_totalSize3D);
			VectorType	m_inProjTwoSub(m_totalSize3D);
			m_inProjOneSub.fill(0.);
			m_inProjTwoSub.fill(0.);
			
			m_inProjOneSub = forwardProjectedVectorOne - this->m_inProjOne;
			m_inProjTwoSub = forwardProjectedVectorTwo - this->m_inProjTwo;

			
			// Process the backprojection (A^T (Ax - y_1)) and (A^T (ARx - y_2))
			// assert (!m_inProjOne.is_zero() && !m_inProjTwo.is_zero());
			VectorType	inBackProjOne(m_totalSize3D); 
			VectorType	inBackProjTwo(m_totalSize3D);
			inBackProjOne.fill(0.);
			inBackProjTwo.fill(0.);

			m_MatrixProjector->CalculteMatrixVectorMultiplication(backwardProjectionMatrix, m_inProjOneSub, inBackProjOne);
			m_MatrixProjector->CalculteMatrixVectorMultiplication(backwardProjectionMatrix, m_inProjTwoSub, inBackProjTwo);

			// Obtain the transpose of affine transformation matrix with the backprojection set two (R^T A^T (ARx - y_2))
			// assert (!inBackProjOne.is_zero() && !inBackProjTwo.is_zero());
			VectorType	inAffineTransposeBackProjTwo(m_totalSize3D);
			inAffineTransposeBackProjTwo.fill(0.);
			
			m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrixTranspose, inBackProjTwo, inAffineTransposeBackProjTwo);


			// Update the derivative with respect to voxel values x by using (A^T (Ax - y_1) + R^T A^T (ARx - y_2))
			derivative.update((inBackProjOne + inAffineTransposeBackProjTwo), 0);


			// Get the derivative with respect to the transformation parameters which is the cross product as ((A^T (ARx - y_2))^T (R'x))
			VectorType	outputVectorGradTemp(m_totalSize3D);
			VectorType  derivativeParameters(m_paraNumber);
			outputVectorGradTemp.fill(0.);
			derivativeParameters.fill(0.);
			
			// Calculate the derivative for the transformation parameter
			assert( !inBackProjTwo.is_zero() );

			FullMatrixType jacobianMatrixFull(m_totalSize3D, m_paraNumber, 0.);
			m_AffineTransformer->CalculteMatrixVectorGradient(jacobianMatrixFull, inVolumeSize, pEulerAffineParameters);

			derivativeParameters = inBackProjTwo.post_multiply(jacobianMatrixFull);

			std::cerr << "Backprojection size: "  << inBackProjTwo.size() 				<< std::endl;
			std::cerr << "Gradient vector size: " << outputVectorGradTemp.size() 	<< std::endl;

			std::ofstream inBackProjTwoFile("inBackProjTwoFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	inBackProjTwoFile << inBackProjTwo << std::endl;

			std::ofstream outputVectorGradTempFile("outputVectorGradTempFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	outputVectorGradTempFile << outputVectorGradTemp << std::endl;

			std::ofstream derivativeParametersFile("derivativeParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	derivativeParametersFile << derivativeParameters << std::endl;

			// Update the derivative with respect to transformation parameters by using ((A^T (ARx - y_2))^T (R'x))
			derivative.update(derivativeParameters, m_totalSize3D);

			std::ofstream derivativeVectorFile("derivativeVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	derivativeVectorFile << derivative << std::endl;		

		}
		*/

  /* -----------------------------------------------------------------------
     GetValueAndDerivative() - Get both the value and derivative of the metric
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>
    ::GetValueAndDerivative(const ParametersType &parameters, 
        MeasureType &Value, DerivativeType &Derivative) const
    {
	  niftkitkDebugMacro(<< "SimultaneousUnconstrainedMatrixReconRegnMetric<TScalarType, IntensityType>::GetValueAndDerivative()");

      // Compute the similarity

      Value = this->GetValue( parameters );

      // Compute the derivative

      this->GetDerivative( parameters, Derivative );
    }

} // end namespace itk


#endif
