/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSimultaneousUnconstrainedMatrixReconRegnMetric_h
#define itkSimultaneousUnconstrainedMatrixReconRegnMetric_h

#include <itkConceptChecking.h>
#include <itkSingleValuedCostFunction.h>

#include <itkForwardAndBackwardProjectionMatrix.h>
#include <itkEulerAffineTransformMatrixAndItsVariations.h>



namespace itk
{
  
/** \class SimultaneousUnconstrainedMatrixReconRegnMetric
 * \brief Class to compute the difference between a reconstruction
 * estimate and the target set of 2D projection images.
 * 
 * This is essentially the ForwardProjectionWithAffineTransformDifferenceFilter
 * repackaged as an ITK cost function.
 */

template <class TScalarType = double, class IntensityType = float>
class ITK_EXPORT SimultaneousUnconstrainedMatrixReconRegnMetric : public SingleValuedCostFunction
{
public:

  /** Standard class typedefs. */
  typedef SimultaneousUnconstrainedMatrixReconRegnMetric    Self;
  typedef SingleValuedCostFunction     											Superclass;
  typedef SmartPointer<Self>           											Pointer;
  typedef SmartPointer<const Self>     											ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimultaneousUnconstrainedMatrixReconRegnMetric, SingleValuedCostFunction);

  // Some convenient typedefs.

	typedef itk::ForwardAndBackwardProjectionMatrix< TScalarType, IntensityType > 								MatrixProjectorType;
	typedef typename MatrixProjectorType::Pointer 																								MatrixProjectorPointerType;

  typedef typename MatrixProjectorType::InputImageType    																			InputVolumeType;
  typedef typename MatrixProjectorType::InputImagePointer 																			InputVolumePointer;
	typedef typename MatrixProjectorType::InputImageConstPointer 																	InputVolumeConstPointer;
	typedef typename MatrixProjectorType::InputImageRegionType   																	InputVolumeRegionType;
	typedef typename MatrixProjectorType::InputImageSizeType   																		InputVolumeSizeType;
	typedef typename MatrixProjectorType::InputImageSpacingType   																InputVolumeSpacingType;
	typedef typename MatrixProjectorType::InputImagePointType   																	InputVolumePointType;
	typedef typename MatrixProjectorType::InputImageIndexType																			InputVolumeIndexType;

  typedef typename MatrixProjectorType::OutputImageType    																			InputProjectionType;
  typedef typename MatrixProjectorType::OutputImagePointer 																			InputProjectionPointer;
	typedef typename MatrixProjectorType::OutputImageConstPointer 																InputProjectionConstPointer;
	typedef typename MatrixProjectorType::OutputImageSizeType																			InputProjectionSizeType;

  typedef itk::EulerAffineTransformMatrixAndItsVariations< double >			 												AffineTransformerType;
  typedef AffineTransformerType::EulerAffineTransformType 																			EulerAffineTransformType;

	typedef itk::ProjectionGeometry< float > 																											ProjectionGeometryType;

  /** Create a sparse matrix to store the affine transformation matrix coefficients */
  typedef vnl_sparse_matrix<TScalarType>           								SparseMatrixType;
  typedef vnl_matrix<TScalarType>           											FullMatrixType;
  typedef vnl_vector<TScalarType>                   							VectorType;

  /**  Type of the parameters. */
  typedef typename SingleValuedCostFunction::ParametersType 			ParametersType;
  typedef typename SingleValuedCostFunction::MeasureType    			MeasureType;
  typedef typename SingleValuedCostFunction::DerivativeType 			DerivativeType;


  /// Set the 3D reconstruction estimate input volume
  void SetInputVolume( InputVolumePointer inVolume ) { m_inVolume = inVolume; }

  /// Set the 3D reconstruction estimate input volume as a vector form
  void SetInputVolumeVector( VectorType &inVolumeVector ) { m_EstimatedVolumeVector = inVolumeVector; }

  /// Set the input projection images
  void SetInputTwoProjectionVectors( VectorType &inProjectionOne, VectorType &inProjectionTwo ) 
	{ m_inProjOne = inProjectionOne; m_inProjTwo = inProjectionTwo; }

  /// Set the temporary projection image
  void SetInputTempProjections( InputProjectionPointer tempProjection ) { m_inProjTemp = tempProjection; }

	/// Set the number of the transformation parameters
  void SetParameterNumber( const unsigned int &paraNumber ) { m_paraNumber = paraNumber; }

	/// Set the transformation parameters as a vector form
  void SetParameterVector( VectorType &paraVector ) { m_TransformationParameterVector = paraVector; }

	/// Set the total number of the voxels of the volume
  void SetTotalVoxel( const unsigned long int &totalSize3D ) { m_totalSize3D = totalSize3D; }

	/// Set the total number of the pixels of the projection
  void SetTotalPixel( const unsigned long int &totalSize2D ) { m_totalSize2D = totalSize2D; }

	/// Set the total number of the pixels of the projection
  void SetTotalProjectionNumber( const unsigned int &projNumber ) { m_ProjectionNumber = projNumber; }

	/// Set the total number of the pixels of the projection
  void SetTotalProjectionSize( InputProjectionSizeType &projSize ) { m_InProjectionSize = projSize; }

	/// Set the projection geometry
  void SetProjectionGeometry( ProjectionGeometryType::Pointer pGeometry ) { m_Geometry = pGeometry; }

  /// Set the size, resolution and origin of the input volume
  void SetInputVolumeSize(InputVolumeSizeType &inVolumeSize) {m_InVolumeSize = inVolumeSize;};
  void SetInputVolumeSpacing(InputVolumeSpacingType &inVolumeSpacing) {m_InVolumeSpacing = inVolumeSpacing;};
  void SetInputVolumeOrigin(InputVolumePointType &inVolumeOrigin) {m_InVolumeOrigin = inVolumeOrigin;};

  /** Return the number of parameters required by the Transform */
  unsigned int GetNumberOfParameters(void) const;

  /** Initialise the metric */
  void Initialise(void);

  /** This method returns the value and derivative of the cost function corresponding
    * to the specified parameters    */ 
  virtual void GetValueAndDerivative( const ParametersType & parameters,
                                      MeasureType & value,
                                      DerivativeType & derivative ) const;

  /** This method returns the value of the cost function corresponding
    * to the specified parameters. This method set to protected
    * to test whether the optimizer only ever calls
    * GetValueAndDerivative() which case we can get away without
    * performing the forward and back-projections for both GetValue()
    * and GetDerivative(). */ 
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** This method returns the derivative of the cost function corresponding
    * to the specified parameters. This method set to protected
    * to test whether the optimizer only ever calls
    * GetValueAndDerivative() which case we can get away without
    * performing the forward and back-projections for both GetValue()
    * and GetDerivative().  */ 
  virtual void GetDerivative( const ParametersType & parameters,
                              DerivativeType & derivative ) const;


protected:

  SimultaneousUnconstrainedMatrixReconRegnMetric();
  virtual ~SimultaneousUnconstrainedMatrixReconRegnMetric() {};

  void PrintSelf(std::ostream& os, Indent indent) const;

	/// Vectors of the image and transformation parameters
	VectorType 																									m_EstimatedVolumeVector;
	VectorType 																									m_TransformationParameterVector;
	VectorType																									m_inProjOne, m_inProjTwo;

	MatrixProjectorPointerType																	m_MatrixProjector;
	AffineTransformerType::Pointer 															m_AffineTransformer;

	InputVolumePointer																					m_inVolume;
	InputProjectionPointer																			m_inProjTemp;

	unsigned int 																								m_paraNumber;
	unsigned long int																						m_totalSize3D;
	unsigned long int																						m_totalSize2D;
	unsigned int 																								m_ProjectionNumber;

  InputVolumeSizeType 																				m_InVolumeSize;
  InputVolumeSpacingType 																			m_InVolumeSpacing;
  InputVolumePointType 																				m_InVolumeOrigin;

	InputProjectionSizeType                                     m_InProjectionSize;

	
	/// Here is problemtic: How to allocate a member variable to store and update the 12 affine parameters?
	/// All the attempts are failed at the moment.
	// EulerAffineTransformType::ParametersType 									m_EulerAffineParameters;
	// double* 																										pEulerAffineParameters;

	ProjectionGeometryType::Pointer 														m_Geometry; 


private:
  SimultaneousUnconstrainedMatrixReconRegnMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimultaneousUnconstrainedMatrixReconRegnMetric.txx"
#endif

#endif
