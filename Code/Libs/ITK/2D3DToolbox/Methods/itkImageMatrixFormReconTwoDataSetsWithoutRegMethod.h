/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkImageMatrixFormReconTwoDataSetsWithoutRegMethod_h
#define itkImageMatrixFormReconTwoDataSetsWithoutRegMethod_h

#include <itkProcessObject.h>
#include <itkImage.h>
#include <itkImageMatrixFormReconTwoDataSetsWithoutRegMetric.h>
#include <itkSingleValuedNonLinearOptimizer.h>
#include <itkProjectionGeometry.h>
#include <itkReconstructionUpdateCommand.h>

namespace itk
{

/** \class ImageMatrixFormReconTwoDataSetsWithoutRegMethod
 * \brief Base class for Image Reconstruction Methods
 *
 * This class defines the generic interface for a reconstruction method.
 *
 * This class is templated over the type of the images to be
 * reconstructed. 
 *
 * The method uses a generic optimizer that can
 * be selected at run-time. The only restriction for the optimizer is
 * that it should be able to operate in single-valued cost functions
 * given that the metrics used to compare images provide a single 
 * value as output.
 */
template <class IntensityType = double>
class ITK_EXPORT ImageMatrixFormReconTwoDataSetsWithoutRegMethod : public ProcessObject 
{
public:
  /** Standard class typedefs. */
  typedef ImageMatrixFormReconTwoDataSetsWithoutRegMethod  					Self;
  typedef ProcessObject  																						Superclass;
  typedef SmartPointer<Self>   																			Pointer;
  typedef SmartPointer<const Self>  																ConstPointer;

  typedef itk::ReconstructionUpdateCommand     											ReconstructionUpdateCommandType;
  typedef ReconstructionUpdateCommand::Pointer 											ReconstructionUpdateCommandPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageMatrixFormReconTwoDataSetsWithoutRegMethod, ProcessObject);

  // Some convenient typedefs.

  /** Intensity type has to be double because the optimizer expects
  the parameters (intensities) to be double */

  typedef Image<IntensityType, 3>                         							MatrixFormReconstructionType;
  typedef typename MatrixFormReconstructionType::Pointer     						MatrixFormReconstructionPointer;
  typedef typename MatrixFormReconstructionType::RegionType  						MatrixFormReconstructionRegionType;
  typedef typename MatrixFormReconstructionType::PixelType    					MatrixFormReconstructionPixelType;
  typedef typename MatrixFormReconstructionType::SizeType    						MatrixFormReconstructionSizeType;
  typedef typename MatrixFormReconstructionType::SpacingType 						MatrixFormReconstructionSpacingType;
  typedef typename MatrixFormReconstructionType::PointType   						MatrixFormReconstructionPointType;
	typedef typename MatrixFormReconstructionType::IndexType    					MatrixFormReconstructionIndexType;

  /// Type of the optimizer.
  typedef SingleValuedNonLinearOptimizer           											OptimizerType;
  typedef typename    OptimizerType::Pointer       											OptimizerPointer;

  /// The type of the metric
  typedef ImageMatrixFormReconTwoDataSetsWithoutRegMetric<double>				MetricType;
  typedef typename MetricType::Pointer          												MetricPointer;

  /// The projection geometry type
  typedef itk::ProjectionGeometry<double> 			  											ProjectionGeometryType;
  typedef typename ProjectionGeometryType::Pointer 											ProjectionGeometryPointer;

  /** Type of the optimisation parameters (reconstructed intensities).
   *  This is the same type used to represent the search space of the
   *  optimization algorithm */
  typedef typename MetricType::ParametersType    												ParametersType;

  /** Type for the output: Using Decorator pattern for enabling
   *  the reconstructed volume to be passed in the data pipeline */
  typedef MatrixFormReconstructionType                     							MatrixFormReconstructionOutputType;
  typedef typename MatrixFormReconstructionOutputType::Pointer       		MatrixFormReconstructionOutputPointer;
  typedef typename MatrixFormReconstructionOutputType::ConstPointer  		MatrixFormReconstructionOutputConstPointer;
  
  /** Set/Get the Optimizer. */
  itkSetObjectMacro( Optimizer,  OptimizerType );
  itkGetObjectMacro( Optimizer,  OptimizerType );

  /** Set/Get the Metric. */
  itkSetObjectMacro( Metric, MetricType );
  itkGetObjectMacro( Metric, MetricType );

  /** Set/Get the Projection Geometry. */
  itkSetObjectMacro( ProjectionGeometry, ProjectionGeometryType );
  itkGetObjectMacro( ProjectionGeometry, ProjectionGeometryType );

  /** Set/Get the ReconstructionUpdateCommand. */
  itkSetObjectMacro( ReconstructionUpdateCommand, ReconstructionUpdateCommandType );
  itkGetObjectMacro( ReconstructionUpdateCommand, ReconstructionUpdateCommandType );

  /// Set the 3D reconstruction estimate volume 
  void SetReconEstimate( MatrixFormReconstructionType *im3D);

  /// Update the 3D reconstruction estimate volume 
  void UpdateReconstructionEstimate( MatrixFormReconstructionType *im3D);
  /** Update the 3D reconstruction estimate volume with the average of
      the existing estimate and the supplied volume. */

  /// Update the initial optimisation parameters
  void UpdateInitialParameters(void);

#if 0

  void UpdateReconstructionEstimateWithAverage( MatrixFormReconstructionType *im3D);

  /// Set the input volume of projection images
  bool SetInputProjectionVolume( MatrixFormReconstructionType *im2D);
#endif

  /// Set the size, resolution and origin of the reconstructed image
  void SetMatrixFormReconstructedVolumeSize(MatrixFormReconstructionSizeType &reconSize) {m_ReconstructedVolumeSize = reconSize;};
  void SetMatrixFormReconstructedVolumeSpacing(MatrixFormReconstructionSpacingType &reconSpacing) {m_ReconstructedVolumeSpacing = reconSpacing;};
  void SetMatrixFormReconstructedVolumeOrigin(MatrixFormReconstructionPointType &reconOrigin) {m_ReconstructedVolumeOrigin = reconOrigin;};

  /** Initialise by setting the interconnects between the components. */
  virtual void Initialise() throw (ExceptionObject);

#if 0
  /** Returns the input image  */
  MatrixFormReconstructionType *GetInput();
#endif

  /** Returns the image resulting from the reconstruction process  */
  MatrixFormReconstructionOutputType *GetOutput();

  /** Returns the image resulting from the reconstruction process  */
  MatrixFormReconstructionOutputType *GetReconstructedVolume() const;

  /** Make a DataObject of the correct type to be used as the specified
   * output. */
  virtual DataObjectPointer MakeOutput(unsigned int idx);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars */
  unsigned long GetMTime() const;  
    
protected:
  ImageMatrixFormReconTwoDataSetsWithoutRegMethod();
  virtual ~ImageMatrixFormReconTwoDataSetsWithoutRegMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** We avoid propagating the input region to the output by
  overloading this function */
  virtual void GenerateOutputInformation() {};
  
  /** Method that initiates the reconstruction. This will Initialise and ensure
   * that all inputs the registration needs are in place, via a call to 
   * Initialise() will then start the optimization process via a call to 
   * StartOptimization()  */
  void StartMatrixFormReconstruction(void);

  /** Method that initiates the optimization process. This method should not be
   * called directly by the users. Instead, this method is intended to be
   * invoked internally by the StartReconstruction() which is in turn invoked by
   * the Update() method. */
  void StartOptimization(void);

  /** Method invoked by the pipeline in order to trigger the computation of 
   * the reconstruction. */
  void  GenerateData ();


private:
  ImageMatrixFormReconTwoDataSetsWithoutRegMethod(const Self&); // purposely not implemented
  void operator=(const Self&);	          // purposely not implemented
  
  bool                             						m_FlagInitialised;

  OptimizerPointer                 						m_Optimizer;
  MetricPointer                    						m_Metric;
  ProjectionGeometryPointer        						m_ProjectionGeometry;

  MatrixFormReconstructionPointer            	m_VolumeEstimate;

  ParametersType                   						m_InitialParameters;
  ParametersType                   						m_LastParameters;
    
  MatrixFormReconstructionSizeType           	m_ReconstructedVolumeSize;
  MatrixFormReconstructionSpacingType        	m_ReconstructedVolumeSpacing;
  MatrixFormReconstructionPointType          	m_ReconstructedVolumeOrigin;

  /** To print out the reconstruction status as we go. */
  ReconstructionUpdateCommandPointer 					m_ReconstructionUpdateCommand;

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageMatrixFormReconTwoDataSetsWithoutRegMethod.txx"
#endif

#endif




