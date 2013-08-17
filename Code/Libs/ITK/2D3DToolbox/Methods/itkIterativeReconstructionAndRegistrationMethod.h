/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkIterativeReconstructionAndRegistrationMethod_h
#define itkIterativeReconstructionAndRegistrationMethod_h

#include <itkProcessObject.h>
#include "itkImageReconstructionMethod.h"
#include <itkImageRegistrationFilter.h>
#include <itkDataObjectDecorator.h>
#include <itkReconstructionAndRegistrationUpdateCommand.h>


namespace itk
{

/** \class IterativeReconstructionAndRegistrationMethod
 * \brief Base class for iterative image reconstruction and registration methods
 *
 * This class defines the generic interface for an iterative image
 * reconstruction and registration method.
 *
 * This class is templated over the type of the images to be
 * reconstructed and registered. 
 */
template <class IntensityType = double>
class ITK_EXPORT IterativeReconstructionAndRegistrationMethod : public ProcessObject 
{
public:
  /** Standard class typedefs. */
  typedef IterativeReconstructionAndRegistrationMethod  Self;
  typedef ProcessObject  Superclass;
  typedef SmartPointer<Self>   Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  typedef itk::ReconstructionAndRegistrationUpdateCommand     ReconstructionAndRegistrationUpdateCommandType;
  typedef ReconstructionAndRegistrationUpdateCommand::Pointer ReconstructionAndRegistrationUpdateCommandPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(IterativeReconstructionAndRegistrationMethod, ProcessObject);

  /// Define the reconstruction types

  typedef itk::ImageReconstructionMethod<IntensityType> ImageReconstructionMethodType;

  typedef typename ImageReconstructionMethodType::Pointer                   ImageReconstructionMethodPointer;
  typedef typename ImageReconstructionMethodType::InputProjectionVolumeType InputProjectionVolumeType;
  typedef typename ImageReconstructionMethodType::ReconstructionType        ReconstructionType;

  typedef typename ReconstructionType::RegionType  ReconstructionRegionType;
  typedef typename ReconstructionType::SizeType    ReconstructionSizeType;
  typedef typename ReconstructionType::IndexType   ReconstructionIndexType;
  typedef typename ReconstructionType::SpacingType ReconstructionSpacingType;
  typedef typename ReconstructionType::PointType   ReconstructionPointType;
  typedef typename ReconstructionType::Pointer     ReconstructionPointer;

  /// Type of the optimizer.
  typedef typename ImageReconstructionMethodType::OptimizerType          ReconstructionOptimizerType;
  typedef typename ReconstructionOptimizerType::Pointer                  ReconstructionOptimizerPointer;

  /// The type of the metric
  typedef typename ImageReconstructionMethodType::MetricType             ReconstructionMetricType;
  typedef typename ReconstructionMetricType::Pointer                     ReconstructionMetricPointer;

  /// The projection geometry type
  typedef typename ImageReconstructionMethodType::ProjectionGeometryType ProjectionGeometryType;
  typedef typename ProjectionGeometryType::Pointer                       ProjectionGeometryPointer;

  /// Define the registration filter type
  
  typedef itk::ImageRegistrationFilter<ReconstructionType, ReconstructionType, 3, double, float>  RegistrationFilterType;  

  typedef typename RegistrationFilterType::Pointer       ImageRegistrationFilterPointer;
  typedef typename RegistrationFilterType::TransformType TransformType;


  /** Type for the output: Using Decorator pattern for enabling
   *  the reconstructed volume to be passed in the data pipeline */
  typedef ReconstructionType                                              ReconstructionAndRegistrationOutputType;
  typedef typename ReconstructionAndRegistrationOutputType::Pointer       ReconstructionAndRegistrationOutputPointer;
  typedef typename ReconstructionAndRegistrationOutputType::ConstPointer  ReconstructionAndRegistrationOutputConstPointer;
  
  /** Type for the output: Using Decorator pattern for enabling
   *  the Transform to be passed in the data pipeline */
  typedef  DataObjectDecorator< TransformType >    TransformOutputType;
  typedef typename TransformOutputType::Pointer    TransformOutputPointer;
  typedef typename TransformOutputType::ConstPointer    TransformOutputConstPointer;
  
  /** Set/Get the ImageRegistrationFilter. */
  itkSetObjectMacro( RegistrationFilter, RegistrationFilterType );
  itkGetObjectMacro( RegistrationFilter, RegistrationFilterType );

  /** Set/Get the 'update 3D reconstruction estimate volume with average' flag */
  void SetFlagUpdateReconEstimateWithAverage( bool flag) {m_FlagUpdateReconEstimateWithAverage = flag; this->Modified();}
  bool GetFlagUpdateReconEstimateWithAverage( void ) {return m_FlagUpdateReconEstimateWithAverage;}

  /** Set/Get the ReconstructionAndRegistrationUpdateCommand. */
  itkSetObjectMacro( ReconstructionAndRegistrationUpdateCommand, ReconstructionAndRegistrationUpdateCommandType );
  itkGetObjectMacro( ReconstructionAndRegistrationUpdateCommand, ReconstructionAndRegistrationUpdateCommandType );

  /// Set the fixed image volume of projection images
  void SetInputFixedImageProjections( InputProjectionVolumeType *imFixedProjections);
  /// Set the moving image volume of projection images
  void SetInputMovingImageProjections( InputProjectionVolumeType *imMovingProjections);

  /// Set the fixed image 3D reconstruction estimate volume 
  void SetFixedReconEstimate( ReconstructionType *im3D);
  /// Set the moving image 3D reconstruction estimate volume 
  void SetMovingReconEstimate( ReconstructionType *im3D);

  /// Set the size, resolution and origin of the reconstructed image
  void SetReconstructedVolumeSize(ReconstructionSizeType &reconSize);
  void SetReconstructedVolumeSpacing(ReconstructionSpacingType &reconSpacing);
  void SetReconstructedVolumeOrigin(ReconstructionPointType &reconOrigin);

  /// Set the Projection Geometry.
  void SetProjectionGeometry(ProjectionGeometryType *geometry) {
    m_FixedImageReconstructor->SetProjectionGeometry(geometry);
    m_MovingImageReconstructor->SetProjectionGeometry(geometry);
    this->Modified();
  }

  /// Set the fixed image reconstruction metric
  void SetFixedReconstructionMetric(ReconstructionMetricType *metric) {
    m_FixedImageReconstructor->SetMetric(metric);
    this->Modified();
  }
  /// Set the moving image reconstruction metric
  void SetMovingReconstructionMetric(ReconstructionMetricType *metric) {
    m_MovingImageReconstructor->SetMetric(metric);
    this->Modified();
  }

  /// Set the fixed image reconstruction optimizer
  void SetFixedReconstructionOptimizer(ReconstructionOptimizerType *optimizer) {
    m_FixedImageReconstructor->SetOptimizer(optimizer);
    this->Modified();
  }
  /// Set the moving image reconstruction optimizer
  void SetMovingReconstructionOptimizer(ReconstructionOptimizerType *optimizer) {
    m_MovingImageReconstructor->SetOptimizer(optimizer);
    this->Modified();
  }

  /// Set the number of combined registration-reconstruction iterations to perform
  void SetNumberOfReconRegnIterations(unsigned int n) {
    m_NumberOfReconRegnIterations = n;
    this->Modified();
  }
  /// Get the number of combined registration-reconstruction iterations to perform
  void GetNumberOfReconRegnIterations(void) {
    return m_NumberOfReconRegnIterations;
  }

  /** Initialise by setting the interconnects between the components. */
  virtual void Initialise() throw (ExceptionObject);

  /** Returns the result of the image reconstruction */
  const ReconstructionType* GetReconOutput(unsigned int output) const;
  /** Returns the result of the fixed image reconstruction */
  const ReconstructionType* GetFixedReconOutput(void) const {return GetReconOutput(0);}
  /** Returns the result of the moving image reconstruction */
  const ReconstructionType* GetMovingReconOutput(void) const {return GetReconOutput(1);}

  /** Returns the transformation */
  const TransformType* GetTransformationOutput(void) const;

  /** Returns the image resulting from the reconstruction process  */
  const ReconstructionAndRegistrationOutputType *GetReconstructedVolume() const;

  /// Graft an object onto the output of this class
  void GraftNthOutput(unsigned int idx, const itk::DataObject *graft);

  /** Method to return the latest modified time of this object or
   * any of its cached ivars */
  unsigned long GetMTime() const;  
    
protected:
  IterativeReconstructionAndRegistrationMethod();
  virtual ~IterativeReconstructionAndRegistrationMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** We avoid propagating the input region to the output by
  overloading this function */
  virtual void GenerateOutputInformation() {};
  
  /** Method that initiates the reconstruction and registration. This will ensure
   * that all inputs the method needs are in place, via a call to 
   * Initialise() and will then start the optimization process via a call to 
   * StartOptimization()  */
  void StartReconstructionAndRegistration(void);

  /** Method that initiates the optimization process. This method should not be
   * called directly by the users. Instead, this method is intended to be
   * invoked internally by the StartReconstructionAndRegistration() which is in turn invoked by
   * the Update() method. */
  void StartOptimization(void);

  /** Method invoked by the pipeline in order to trigger the computation of 
   * the reconstruction. */
  void  GenerateData ();


private:
  IterativeReconstructionAndRegistrationMethod(const Self&); // purposely not implemented
  void operator=(const Self&);                               // purposely not implemented

  bool m_FlagInitialised;

  /* Flag indicating whether to update the 3D reconstruction estimate volume with the average of
      the existing estimate and the supplied volume. */
  bool m_FlagUpdateReconEstimateWithAverage;

  // The number of combined registration-reconstruction iterations to perform
  unsigned int m_NumberOfReconRegnIterations;

  // Pointers to the source and target reconstructors
  ImageReconstructionMethodPointer m_FixedImageReconstructor;
  ImageReconstructionMethodPointer m_MovingImageReconstructor;

  // Pointer to the registration filter
  ImageRegistrationFilterPointer m_RegistrationFilter;

  /** To print out the reconstruction and registration status as we go. */
  ReconstructionAndRegistrationUpdateCommandPointer m_ReconstructionAndRegistrationUpdateCommand;

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIterativeReconstructionAndRegistrationMethod.txx"
#endif

#endif



