/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkImageReconstructionMetric_h
#define __itkImageReconstructionMetric_h

#include "itkImageToImageFilter.h"
#include "itkConceptChecking.h"
#include "itkSingleValuedCostFunction.h"
#include "itkForwardAndBackProjectionDifferenceFilter.h"


namespace itk
{
  
/** \class ImageReconstructionMetric
 * \brief Class to compute the difference between a reconstruction
 * estimate and the target set of 2D projection images.
 * 
 * This is essentially the ForwardAndBackProjectionDifferenceFilter
 * repackaged as an ITK cost function.
 */

template <class IntensityType = float>
class ITK_EXPORT ImageReconstructionMetric : public SingleValuedCostFunction
{
public:

  /** Standard class typedefs. */
  typedef ImageReconstructionMetric    Self;
  typedef SingleValuedCostFunction     Superclass;
  typedef SmartPointer<Self>           Pointer;
  typedef SmartPointer<const Self>     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageReconstructionMetric, ImageToImageFilter);

  /** Some convenient typedefs. */

  /** The ImageToImageFilter to perform the forward and back-projection */
  typedef itk::ForwardAndBackProjectionDifferenceFilter<IntensityType> ForwardAndBackProjectionDifferenceFilterType;
  typedef typename ForwardAndBackProjectionDifferenceFilterType::Pointer ForwardAndBackProjectionDifferenceFilterPointerType;

  typedef typename ForwardAndBackProjectionDifferenceFilterType::InputVolumeType    InputVolumeType;
  typedef typename ForwardAndBackProjectionDifferenceFilterType::InputVolumePointer InputVolumePointer;

  typedef typename ForwardAndBackProjectionDifferenceFilterType::InputProjectionVolumeType InputProjectionVolumeType;
  typedef typename ForwardAndBackProjectionDifferenceFilterType::InputProjectionVolumePointer InputProjectionVolumePointer;

  typedef typename ForwardAndBackProjectionDifferenceFilterType::ProjectionGeometryType ProjectionGeometryType;
  typedef typename ForwardAndBackProjectionDifferenceFilterType::ProjectionGeometryPointer ProjectionGeometryPointer;

  typedef typename ForwardAndBackProjectionDifferenceFilterType::OutputBackProjectedDifferencesType OutputBackProjectedDifferencesType;
  typedef typename ForwardAndBackProjectionDifferenceFilterType::OutputBackProjectedDifferencesPointer OutputBackProjectedDifferencesPointer;
  

  /**  Type of the parameters. */
  typedef typename SingleValuedCostFunction::ParametersType ParametersType;
  typedef typename SingleValuedCostFunction::MeasureType    MeasureType;
  typedef typename SingleValuedCostFunction::DerivativeType DerivativeType;

  /// Set the 3D reconstruction estimate volume input
  void SetInputVolume( InputVolumeType *im3D );

  /// Set the input 3D volume of projection image
  void SetInputProjectionVolume( InputProjectionVolumeType *im2D );

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

  ImageReconstructionMetric();
  virtual ~ImageReconstructionMetric() {};

  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Filename to optionally save the current iteration of the
      reconstruction estimate to */
  std::string fileOutputCurrentEstimate;
  /** Suffix of filename to optionally save the current iteration of the
      reconstruction estimate to */
  std::string suffixOutputCurrentEstimate;

  /// The filter to perform the forward and back-projection
  ForwardAndBackProjectionDifferenceFilterPointerType m_FwdAndBackProjDiffFilter;


private:
  ImageReconstructionMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageReconstructionMetric.txx"
#endif

#endif
