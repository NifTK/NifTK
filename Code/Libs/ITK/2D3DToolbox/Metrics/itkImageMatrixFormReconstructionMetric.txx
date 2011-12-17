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

#ifndef __itkImageMatrixFormReconstructionMetric_txx
#define __itkImageMatrixFormReconstructionMetric_txx

#include "itkImageMatrixFormReconstructionMetric.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ImageMatrixFormReconstructionMetric<IntensityType>
::ImageMatrixFormReconstructionMetric()
{
  // suffixOutputCurrentEstimate = "nii";

  // Create the matrix projector
  m_MatrixProjector = MatrixProjectorType::New();
  
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ImageMatrixFormReconstructionMetric<IntensityType>
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
ImageMatrixFormReconstructionMetric<IntensityType>
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
typename ImageMatrixFormReconstructionMetric<IntensityType>::MeasureType
ImageMatrixFormReconstructionMetric<IntensityType>
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
  VectorType forwardProjectedVector(m_totalSize3D);
  forwardProjectedVector.fill(0.);

  m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVector);

	// Initialise the current measure
  MeasureType currentMeasure;
  currentMeasure = 0.;

	// Calculate (Ax - y_1)
  VectorType	m_inProjSub(m_totalSize3D);
	m_inProjSub.fill(0.);
				
	assert( !(this->m_inProj.is_zero()) );
	m_inProjSub = forwardProjectedVector - this->m_inProj;
			
	std::ofstream estimateVolumeFile("estimateVolumeFile.txt", std::ios::out | std::ios::app | std::ios::binary);
  estimateVolumeFile << m_EstimatedVolumeVector << " ";
		
	// Calculate ||Ax - y_1||^2, which is the cost function value
	currentMeasure = m_inProjSub.squared_magnitude();
	std::cerr << "Current cost function value is: " << currentMeasure << std::endl;

  return currentMeasure;
}


/* -----------------------------------------------------------------------
   GetDerivative() - Get the derivative of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMetric<IntensityType>
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
  VectorType forwardProjectedVector(m_totalSize3D);
  forwardProjectedVector.fill(0.);

  m_MatrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, m_EstimatedVolumeVector, forwardProjectedVector);

	// Calculate (Ax - y_1)
  VectorType	m_inProjSub(m_totalSize3D);
	m_inProjSub.fill(0.);
				
	assert( !(this->m_inProj.is_zero()) );
	m_inProjSub = forwardProjectedVector - this->m_inProj;

	// Process the backprojection (A^T (Ax - y_1))
	VectorType	inBackProj(m_totalSize3D); 
	inBackProj.fill(0.);

	m_MatrixProjector->CalculteMatrixVectorMultiplication(backwardProjectionMatrix, m_inProjSub, inBackProj);

	// std::cerr << "The size of the backprojection is: " 	<< inBackProj.size() << std::endl;
	// std::cerr << "The size of the derivative is: " 			<< derivative.size() << std::endl;

	// derivative.update(inBackProj); // Use this can't update the vector of the derivative!!!?
	derivative = inBackProj;

	// std::cerr << "The size of the derivative is: " 			<< derivative.size() << std::endl;
	
}


/* -----------------------------------------------------------------------
   GetValueAndDerivative() - Get both the value and derivative of the metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMetric<IntensityType>
::GetValueAndDerivative(const ParametersType &parameters, 
                        MeasureType &Value, DerivativeType &Derivative) const
{
  niftkitkDebugMacro(<< "ImageMatrixFormReconstructionMetric<IntensityType>::GetValueAndDerivative()");

  // Compute the similarity

  Value = this->GetValue( parameters );

  // Compute the derivative
  
  this->GetDerivative( parameters, Derivative );
}

} // end namespace itk


#endif
