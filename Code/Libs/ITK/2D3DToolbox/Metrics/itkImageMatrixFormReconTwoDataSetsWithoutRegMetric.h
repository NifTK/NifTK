/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageMatrixFormReconTwoDataSetsWithoutRegMetric_h
#define __itkImageMatrixFormReconTwoDataSetsWithoutRegMetric_h

#include "itkConceptChecking.h"
#include "itkSingleValuedCostFunction.h"

#include "itkForwardAndBackwardProjectionMatrix.h"
#include "itkEulerAffineTransformMatrixAndItsVariations.h"


namespace itk
{
  
/** \class ImageMatrixFormReconTwoDataSetsWithoutRegMetric
 * \brief Class to compute the difference between a reconstruction
 * estimate and the target set of 2D projection images.
 * 
 * This is essentially the ForwardAndBackProjectionDifferenceFilter
 * repackaged as an ITK cost function.
 */

template <class IntensityType = double>
class ITK_EXPORT ImageMatrixFormReconTwoDataSetsWithoutRegMetric : public SingleValuedCostFunction
{
public:

  /** Standard class typedefs. */
  typedef ImageMatrixFormReconTwoDataSetsWithoutRegMetric    	Self;
  typedef SingleValuedCostFunction     												Superclass;
  typedef SmartPointer<Self>           												Pointer;
  typedef SmartPointer<const Self>     												ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageMatrixFormReconTwoDataSetsWithoutRegMetric, SingleValuedCostFunction);

  /** Some convenient typedefs. */
  typedef vnl_vector<double>               															VectorType;
	typedef vnl_sparse_matrix<double>           													SparseMatrixType;

	typedef itk::ForwardAndBackwardProjectionMatrix< double, double > 		MatrixProjectorType;
	typedef typename MatrixProjectorType::Pointer 												MatrixProjectorPointerType;

  typedef typename MatrixProjectorType::InputImageType    							InputVolumeType;
  typedef typename MatrixProjectorType::InputImagePointer 							InputVolumePointer;
	typedef typename MatrixProjectorType::InputImageConstPointer 					InputVolumeConstPointer;
	typedef typename MatrixProjectorType::InputImageRegionType   					InputVolumeRegionType;
	typedef typename MatrixProjectorType::InputImageSizeType   						InputVolumeSizeType;
	typedef typename MatrixProjectorType::InputImageSpacingType   				InputVolumeSpacingType;
	typedef typename MatrixProjectorType::InputImagePointType   					InputVolumePointType;
	typedef typename MatrixProjectorType::InputImageIndexType							InputVolumeIndexType;

  typedef typename MatrixProjectorType::OutputImageType    							InputProjectionType;
  typedef typename MatrixProjectorType::OutputImagePointer 							InputProjectionPointer;
	typedef typename MatrixProjectorType::OutputImageConstPointer 				InputProjectionConstPointer;
	typedef typename MatrixProjectorType::OutputImageSizeType							InputProjectionSizeType;

  typedef itk::EulerAffineTransformMatrixAndItsVariations< double >			AffineTransformerType;
  typedef AffineTransformerType::EulerAffineTransformType 							EulerAffineTransformType;

	typedef itk::ProjectionGeometry< double > 														ProjectionGeometryType;

  /**  Type of the parameters. */
  typedef typename SingleValuedCostFunction::ParametersType 						ParametersType;
  typedef typename SingleValuedCostFunction::MeasureType    						MeasureType;
  typedef typename SingleValuedCostFunction::DerivativeType 						DerivativeType;

  /// Set the 3D reconstruction estimate input volume
  void SetInputVolume( InputVolumePointer inVolume ) { m_inVolume = inVolume; }

  /// Set the 3D reconstruction estimate input volume as a vector form
  void SetInputVolumeVector( VectorType &inVolumeVector ) { m_EstimatedVolumeVector = inVolumeVector; }

  /// Set the input projection images
  void SetInputTwoProjectionVectors( VectorType &inProjectionOne, VectorType &inProjectionTwo ) 
	{ m_inProjOne = inProjectionOne; m_inProjTwo = inProjectionTwo; }

  /// Set the temporary projection image
  void SetInputTempProjections( InputProjectionPointer tempProjection ) { m_inProjTemp = tempProjection; }

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

  /** Return the number of parameters required */
  unsigned int GetNumberOfParameters(void) const;

#if 0
  /// Get the 3D reconstructed volume
  const InputVolumeType *GetReconstructedVolume(void) {
    return m_FwdAndBackProjDiffFilter->GetPointerToInputVolume();
  }

  /** Assign to 'parameters' the address of the
      reconstruction volume estimate voxel intensities. */
  void SetParametersAddress( const ParametersType & parameters ) const;
  /** Assign to 'derivatives' the address of the
      reconstruction volume estimate voxel intensities. */
  void SetDerivativesAddress( const DerivativeType & derivatives ) const;

  /** Set the parameters, i.e. intensities of the reconstruction estimate. */
  void SetParameters( const ParametersType & parameters ) const;

  /** Specify a filename to save the current reconstruction estimate
      at each iteration */
  void SetIterativeReconEstimateFile( std::string filename ) {
    fileOutputCurrentEstimate = filename;
  }

  /** Specify a filename suffix to save the current reconstruction estimate
      at each iteration */
  void SetIterativeReconEstimateSuffix( std::string suffix ) {
    suffixOutputCurrentEstimate = suffix;
  }
#endif

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

  ImageMatrixFormReconTwoDataSetsWithoutRegMetric();
  virtual ~ImageMatrixFormReconTwoDataSetsWithoutRegMetric() {};

  void PrintSelf(std::ostream& os, Indent indent) const;

#if 0
  /** Filename to optionally save the current iteration of the
      reconstruction estimate to */
  std::string fileOutputCurrentEstimate;
  /** Suffix of filename to optionally save the current iteration of the
      reconstruction estimate to */
  std::string suffixOutputCurrentEstimate;
#endif

	/// Vectors of the image
	VectorType 																									m_EstimatedVolumeVector;
	VectorType																									m_inProjOne, m_inProjTwo;

	MatrixProjectorPointerType																	m_MatrixProjector;
	AffineTransformerType::Pointer 															m_AffineTransformer;

	InputVolumePointer																					m_inVolume;
	InputProjectionPointer																			m_inProjTemp;

	unsigned long int																						m_totalSize3D;
	unsigned long int																						m_totalSize2D;
	unsigned int 																								m_ProjectionNumber;

  InputVolumeSizeType 																				m_InVolumeSize;
  InputVolumeSpacingType 																			m_InVolumeSpacing;
  InputVolumePointType 																				m_InVolumeOrigin;

	InputProjectionSizeType                                     m_InProjectionSize;

	ProjectionGeometryType::Pointer 														m_Geometry; 


private:
  ImageMatrixFormReconTwoDataSetsWithoutRegMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageMatrixFormReconTwoDataSetsWithoutRegMetric.txx"
#endif

#endif
