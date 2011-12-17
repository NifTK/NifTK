/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-12-09 15:08:50 +0000 (Thu, 09 Dec 2010) $
 Revision          : $Revision: 4493 $
 Last modified by  : $Author: jhh $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkImageMatrixFormReconTwoDataSetsWithoutRegMetric_txx
#define __itkImageMatrixFormReconTwoDataSetsWithoutRegMetric_txx

#include "itkImageMatrixFormReconTwoDataSetsWithoutRegMetric.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>
::ImageMatrixFormReconTwoDataSetsWithoutRegMetric()
{
  // suffixOutputCurrentEstimate = "nii";

  // Create the matrix projector
  m_MatrixProjector = MatrixProjectorType::New();

  // Allocate the affine transformer
	m_AffineTransformer = AffineTransformerType::New(); 
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
	Superclass::PrintSelf(os,indent);
	os << indent << "CreateForwardBackwardProjectionMatrix: " << std::endl;
}


/* -----------------------------------------------------------------------
   GetNumberOfParameters()
   ----------------------------------------------------------------------- */

template< class IntensityType>
unsigned int 
ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>
::GetNumberOfParameters( void ) const
{
	assert ( m_EstimatedVolumeVector.size()!=0 );
  unsigned int nParameters = m_EstimatedVolumeVector.size();
  return nParameters;
}


/* -----------------------------------------------------------------------
   GetValue() - Get the value of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>::MeasureType
ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>
::GetValue( const ParametersType &parameters ) const
{
	// Get the estimate as a vector
  VectorType m_EstimatedVolumeVector = parameters;
  
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

	// Set the projection geometry
	m_MatrixProjector->SetProjectionGeometry( m_Geometry );

  m_MatrixProjector->GetForwardProjectionSparseMatrix(forwardProjectionMatrix, m_inVolume, m_inProjTemp, 
       inVolumeSize, inProjSize, m_ProjectionNumber);

  // Calculate the matrix/vector multiplication in order to get the forward projection (Ax)
  VectorType forwardProjectedVectorOne(m_totalSize3D);
  forwardProjectedVectorOne.fill(0.);

  m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVectorOne);

	// Create the corresponding transformation matrix
	static SparseMatrixType affineMatrix(m_totalSize3D, m_totalSize3D);

	EulerAffineTransformType::ParametersType fixedCorrectEulerAffineParameters(12);
	fixedCorrectEulerAffineParameters.Fill(0.);

	fixedCorrectEulerAffineParameters.SetElement(0, -10.);
	fixedCorrectEulerAffineParameters.SetElement(2, 20.);

	fixedCorrectEulerAffineParameters.SetElement(4, 30.0);

	fixedCorrectEulerAffineParameters.SetElement(6, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(7, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(8, 1.0);

#if 0
	fixedCorrectEulerAffineParameters.SetElement(0, -1.0);
	fixedCorrectEulerAffineParameters.SetElement(1, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(2, 2.0);

	fixedCorrectEulerAffineParameters.SetElement(3, 10.0);

	fixedCorrectEulerAffineParameters.SetElement(6, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(7, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(8, 1.0);
#endif

	m_AffineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, inVolumeSize, fixedCorrectEulerAffineParameters);

	// Calculate the matrix/vector multiplication in order to get the affine transformation (Rx)
	VectorType affineTransformedVector(m_totalSize3D);
	affineTransformedVector.fill(0.);

	m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, m_EstimatedVolumeVector, affineTransformedVector);

	// Calculate the matrix/vector multiplication in order to get the forward projection (ARx)
	assert (!affineTransformedVector.is_zero());
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

template< class IntensityType>
void
ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>
::GetDerivative( const ParametersType &parameters, 
                 DerivativeType &derivative ) const
{
  // Get the estimate as a vector
  VectorType m_EstimatedVolumeVector = parameters;

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

  // Calculate the matrix/vector multiplication in order to get the forward projection (Ax)
  VectorType forwardProjectedVectorOne(m_totalSize3D);
  forwardProjectedVectorOne.fill(0.);

  m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVectorOne);

	// Create the corresponding transformation matrix
	static SparseMatrixType affineMatrix(m_totalSize3D, m_totalSize3D);
	static SparseMatrixType affineMatrixTranspose(m_totalSize3D, m_totalSize3D);

	EulerAffineTransformType::ParametersType fixedCorrectEulerAffineParameters(12);
	fixedCorrectEulerAffineParameters.Fill(0.);

	fixedCorrectEulerAffineParameters.SetElement(0, -10.);
	fixedCorrectEulerAffineParameters.SetElement(2, 20.);

	fixedCorrectEulerAffineParameters.SetElement(4, 30.0);

	fixedCorrectEulerAffineParameters.SetElement(6, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(7, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(8, 1.0);

#if 0
	fixedCorrectEulerAffineParameters.SetElement(0, -1.0);
	fixedCorrectEulerAffineParameters.SetElement(1, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(2, 2.0);

	fixedCorrectEulerAffineParameters.SetElement(3, 10.0);

	fixedCorrectEulerAffineParameters.SetElement(6, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(7, 1.0);
	fixedCorrectEulerAffineParameters.SetElement(8, 1.0);
#endif

	m_AffineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, inVolumeSize, fixedCorrectEulerAffineParameters);

	// Calculate the matrix/vector multiplication in order to get the affine transformation (Rx)
	VectorType affineTransformedVector(m_totalSize3D);
	affineTransformedVector.fill(0.);

	m_AffineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, m_EstimatedVolumeVector, affineTransformedVector);
  m_AffineTransformer->GetAffineTransformationSparseMatrixT(affineMatrix, affineMatrixTranspose, inVolumeSize);

	// Calculate the matrix/vector multiplication in order to get the forward projection (ARx)
	assert (!affineTransformedVector.is_zero());
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

	// std::cerr << "The size of the backprojection is: " 	<< inBackProj.size() << std::endl;
	// std::cerr << "The size of the derivative is: " 			<< derivative.size() << std::endl;

	// Update the derivative with respect to voxel values x by using (A^T (Ax - y_1) + R^T A^T (ARx - y_2))
	// derivative.update((inBackProjOne + inAffineTransposeBackProjTwo), 0); // Use this can't update the vector of the derivative!!!?
	derivative = inBackProjOne + inAffineTransposeBackProjTwo;

	// std::cerr << "The size of the derivative is: " 			<< derivative.size() << std::endl;
	
}


/* -----------------------------------------------------------------------
   GetValueAndDerivative() - Get both the value and derivative of the metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>
::GetValueAndDerivative(const ParametersType &parameters, 
                        MeasureType &Value, DerivativeType &Derivative) const
{
  niftkitkDebugMacro(<< "ImageMatrixFormReconTwoDataSetsWithoutRegMetric<IntensityType>::GetValueAndDerivative()");

  // Compute the similarity

  Value = this->GetValue( parameters );

  // Compute the derivative
  
  this->GetDerivative( parameters, Derivative );
}

} // end namespace itk


#endif
