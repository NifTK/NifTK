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

#ifndef __itkSimultaneousReconstructionRegistrationMetric_h
#define __itkSimultaneousReconstructionRegistrationMetric_h

#include "itkImageToImageFilter.h"
#include "itkConceptChecking.h"
#include "itkSingleValuedCostFunction.h"

#include "itkForwardProjectionWithAffineTransformDifferenceFilter.h"


namespace itk
{
  
/** \class SimultaneousReconstructionRegistrationMetric
 * \brief Class to compute the difference between a reconstruction
 * estimate and the target set of 2D projection images.
 * 
 * This is essentially the ForwardProjectionWithAffineTransformDifferenceFilter
 * repackaged as an ITK cost function.
 */

template <class IntensityType = float>
class ITK_EXPORT SimultaneousReconstructionRegistrationMetric : public SingleValuedCostFunction
{
public:

  /** Standard class typedefs. */
  typedef SimultaneousReconstructionRegistrationMetric    Self;
  typedef SingleValuedCostFunction     										Superclass;
  typedef SmartPointer<Self>           										Pointer;
  typedef SmartPointer<const Self>     										ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimultaneousReconstructionRegistrationMetric, SingleValuedCostFunction);

  /** Some convenient typedefs. */

  /** The ImageToImageFilter to perform the forward and back-projection */
  typedef itk::ForwardProjectionWithAffineTransformDifferenceFilter<IntensityType> 		ForwardProjectionWithAffineTransformDifferenceFilterType;
  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::Pointer 	ForwardProjectionWithAffineTransformDifferenceFilterPointerType;

  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::InputVolumeType    InputVolumeType;
  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::InputVolumePointer InputVolumePointer;

  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::InputProjectionVolumeType 		InputProjectionVolumeType;
  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::InputProjectionVolumePointer InputProjectionVolumePointer;

  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::ProjectionGeometryType 		ProjectionGeometryType;
  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::ProjectionGeometryPointer 	ProjectionGeometryPointer;

  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::OutputBackProjectedDifferencesType 		OutputBackProjectedDifferencesType;
  typedef typename ForwardProjectionWithAffineTransformDifferenceFilterType::OutputBackProjectedDifferencesPointer 	OutputBackProjectedDifferencesPointer;
  

  /**  Type of the parameters. */
  typedef typename SingleValuedCostFunction::ParametersType ParametersType;
  typedef typename SingleValuedCostFunction::MeasureType    MeasureType;
  typedef typename SingleValuedCostFunction::DerivativeType DerivativeType;

  /// Set the 3D reconstruction estimate volume input
  void SetInputVolume( InputVolumeType *im3D );

  /// Set the input 3D volume of projection image (Two sets of projection images, e.g., y_1 and y_2)
  void SetInputProjectionVolumeOne( InputProjectionVolumeType *im2DNumberOne );
	void SetInputProjectionVolumeTwo( InputProjectionVolumeType *im2DNumberTwo );

  /// Set the projection geometry
  void SetProjectionGeometry( ProjectionGeometryType *projGeometry );

  /** Return the number of parameters required by the Transform */
  unsigned int GetNumberOfParameters(void) const;

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

  /** Initialise the metric */
  void Initialise(void) {m_FwdAndBackProjDiffFilter->Initialise();}

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

  SimultaneousReconstructionRegistrationMetric();
  virtual ~SimultaneousReconstructionRegistrationMetric() {};

  void PrintSelf(std::ostream& os, Indent indent) const;
	
	/// Create and store the Euler affine transform matrix
	ForwardProjectionWithAffineTransformDifferenceFilterPointerType m_FwdAndBackProjDiffFilter;


private:
  SimultaneousReconstructionRegistrationMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimultaneousReconstructionRegistrationMetric.txx"
#endif

#endif
