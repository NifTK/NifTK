/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageRegistrationFactory_h
#define __itkImageRegistrationFactory_h

#include "itkProcessObject.h"
#include "itkConstantBoundaryCondition.h"
#include "itkWindowedSincInterpolateImageFunction.h"

// Interpolators.
#include "itkInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

// Similarity Measures
#include "itkSimilarityMeasure.h"
#include "itkMSDImageToImageMetric.h"
#include "itkNCCImageToImageMetric.h" 
#include "itkSSDImageToImageMetric.h"
#include "itkSADImageToImageMetric.h"
#include "itkRIUImageToImageMetric.h"
#include "itkPIUImageToImageMetric.h"
#include "itkJEImageToImageMetric.h"
#include "itkMIImageToImageMetric.h"
#include "itkNMIImageToImageMetric.h"
#include "itkCRImageToImageMetric.h"

// Transformations
#include "itkTransform.h"
#include "itkPerspectiveProjectionTransform.h" // Unlikely to be optimised in a reg'n but here for completeness
#include "itkEulerAffineTransform.h" // This one does, translations, rotations, rigid, scale, affine
#include "itkBSplineTransform.h"
#include "itkFluidDeformableTransform.h"
#include "itkAffineTransform.h"
#include "itkPCADeformationModelTransform.h"
#include "itkTranslationPCADeformationModelTransform.h"

// Optimizers
#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkConjugateGradientOptimizer.h"
#include "itkPowellOptimizer.h"
#include "itkUCLRegularStepOptimizer.h"
#include "itkUCLPowellOptimizer.h"

// Commands
#include "itkIterationUpdateCommand.h"
#include "itkVnlIterationUpdateCommand.h"

// Registration methods.
#include "itkSingleResolutionImageRegistrationMethod.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkTranslationThenRotationImageRegistrationMethod.h"
#include "itkTranslateRotateScaleImageRegistrationMethod.h"
#include "itkRigidPlusScaleImageRegistrationMethod.h"
#include "itkBlockMatchingMethod.h"

// Multi-resolution methods
#include "itkMultiResolutionImageRegistrationWrapper.h"

namespace itk
{

enum 
InterpolationTypeEnum
{
  UNKNOWN_INTERP,
  NEAREST,
  LINEAR,
  BSPLINE,
  SINC 
};

enum 
MetricTypeEnum
{
  UNKNOWN_METRIC,
  SSD,            // Sum of Squared Differences
  MSD,            // Mean of Squared Differences
  SAD,            // Sum of Absolute Differences
  NCC,            // Normalized Cross Correlation
  RIU,            // Woods Ratio Image Uniformity
  PIU,            // Woods Partitioned Image Uniformity
  JE,             // Joint Entropy
  MI,             // Mutual information
  NMI,            // Normalized Mutual Information
  CR              // Correlation Ratio
};

enum
TransformTypeEnum
{
  UNKNOWN_TRANSFORM,
  TRANSLATION,    // Mainly for testing.
  RIGID,          // Rigid, so rotations and translations, 3DOF in 2D and 6DOF in 3D.
  RIGID_SCALE,    // Rigid plus scale, 5DOF in 2D, 9DOF in 3D.
  AFFINE          // Affine. 7DOF in 2D, 12DOF in 3D.
};

enum
OptimizerTypeEnum
{
  UNKNOWN_OPTIMIZER,
  SIMPLEX,                    // For when you dont trust, or don't have derivative.
  GRADIENT_DESCENT,           // Standard gradient descent.
  REGSTEP_GRADIENT_DESCENT,   // Regular step size gradient descent.
  CONJUGATE_GRADIENT_DESCENT, // Conjugate gradients.
  POWELL,                     // also, doesnt require derivative.
  SIMPLE_REGSTEP,             // Simple multi-regular step in each direction.
  UCLPOWELL
};

enum
SingleResRegistrationMethodTypeEnum
{
  UNKNOWN_METHOD,
  SINGLE_RES_MASKED,             // The 'default' method, simply optimises transform wrt metric.
  SINGLE_RES_TRANS_ROTATE,       // Switching method, just does rigid, but alternates translation and rotation
  SINGLE_RES_TRANS_ROTATE_SCALE, // Switching method, separately does translation, rotation and scale
  SINGLE_RES_RIGID_SCALE,        // Switching method, alternates rigid, and then scale.
  SINGLE_RES_BLOCK_MATCH
};

enum
MultiResRegistrationMethodTypeEnum
{
  UNKNOWN_MULTI,
  MULTI_RES_NORMAL
};
/**
 * \class ImageRegistrationFactory 
 * \brief Parameterised Factory Pattern [2] for creating registration objects.
 * 
 * The purpose of this class is to:
 * 
 * a.) Define the types that we can create, so look at the typedefs below, and enums above.
 * 
 * b.) Set reasonable defaults, if necessary.
 * 
 * i.e. There is NO clever logic here, it's just "how do I create an object"
 * 
 * While there are many ways of doing this, this one was deemed to be the simplest.
 */
template <typename TInputImageType, unsigned int Dimension, class TScalarType>
class ITK_EXPORT ImageRegistrationFactory : public Object 
{
public:
  
  /** Standard class typedefs. */
  typedef ImageRegistrationFactory  Self;
  typedef Object                    Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRegistrationFactory, Object);

  /** Typedefs. */
  typedef typename TInputImageType::PixelType                                       InputPixelType;
  
  /** Iteration Update Commands. */
  typedef itk::IterationUpdateCommand                                               IterationUpdateCommandType;
  typedef itk::VnlIterationUpdateCommand                                            VnlIterationUpdateCommandType;

  /** Interpolators. */
  typedef itk::InterpolateImageFunction< TInputImageType, TScalarType>                InterpolatorType;
  typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, TScalarType> NearestNeighbourInterpolatorType;
  typedef itk::LinearInterpolateImageFunction< TInputImageType, TScalarType >         LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction< TInputImageType, TScalarType >        BSplineInterpolatorType;

  typedef itk::ConstantBoundaryCondition< TInputImageType >                         BoundaryConditionType;
  const static unsigned int                                                         WindowRadius = 5;
  typedef itk::Function::WelchWindowFunction<WindowRadius>  WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction<
                                          TInputImageType,
                                          WindowRadius,
                                          WindowFunctionType,
                                          BoundaryConditionType,
                                          TScalarType  >                            SincInterpolatorType;
                                          
  /** Similarity Measures. We use our base class SimilarityMeasure, not ImageToImageMetric. */
  typedef itk::SimilarityMeasure<TInputImageType, TInputImageType>                  MetricType;
  typedef itk::SSDImageToImageMetric<TInputImageType, TInputImageType >             SSDMetricType;
  typedef itk::MSDImageToImageMetric<TInputImageType, TInputImageType >             MSDMetricType;
  typedef itk::NCCImageToImageMetric<TInputImageType, TInputImageType >             NCCMetricType;  
  typedef itk::SADImageToImageMetric<TInputImageType, TInputImageType >             SADMetricType;    
  typedef itk::RIUImageToImageMetric<TInputImageType, TInputImageType >             RIUMetricType;  
  typedef itk::PIUImageToImageMetric<TInputImageType, TInputImageType >             PIUMetricType;
  typedef itk::JEImageToImageMetric<TInputImageType, TInputImageType >              JEMetricType;
  typedef itk::MIImageToImageMetric<TInputImageType, TInputImageType >              MIMetricType;
  typedef itk::NMIImageToImageMetric<TInputImageType, TInputImageType >             NMIMetricType;
  typedef itk::CRImageToImageMetric<TInputImageType, TInputImageType >              CRMetricType;

  /** Transformations */
  typedef itk::Transform< TScalarType, Dimension, Dimension >                           TransformType;
  typedef itk::PerspectiveProjectionTransform<TScalarType>                              PerspectiveProjectionTransformType;
  typedef itk::EulerAffineTransform<TScalarType, Dimension, Dimension>                  EulerAffineTransformType;
  typedef itk::BSplineTransform<TInputImageType, TScalarType, Dimension, float>         BSplineDeformableTransformType; 
  typedef itk::FluidDeformableTransform<TInputImageType, TScalarType, Dimension, float> FluidDeformableTransformType;
  typedef itk::AffineTransform<TScalarType, Dimension>                                  ITKAffineTransformType;
  typedef itk::PCADeformationModelTransform<TScalarType, Dimension>                     PCADeformationModelTransformType;
  typedef itk::TranslationPCADeformationModelTransform<TScalarType, Dimension>          TranslationPCADeformationModelTransformType;

  /** Optimisers. */
  typedef itk::SingleValuedNonLinearOptimizer                           OptimizerType;
  typedef itk::UCLSimplexOptimizer                                      SimplexType;
  typedef SimplexType*                                                  SimplexPointer;
  typedef itk::GradientDescentOptimizer                                 GradientDescentType;
  typedef GradientDescentType*                                          GradientDescentPointer;
  typedef itk::UCLRegularStepGradientDescentOptimizer                   RegularStepGradientDescentType;
  typedef RegularStepGradientDescentType*                               RegularStepGradientDescentPointer;
  typedef itk::ConjugateGradientOptimizer                               ConjugateGradientType;
  typedef ConjugateGradientType*                                        ConjugateGradientPointer;
  typedef itk::PowellOptimizer                                          PowellOptimizerType;
  typedef PowellOptimizerType*                                          PowellOptimizerPointer;
  typedef itk::UCLRegularStepOptimizer                                  UCLRegularStepOptimizerType;
  typedef UCLRegularStepOptimizerType*                                  UCLRegularStepOptimizerTypePointer;
  typedef itk::UCLPowellOptimizer                                       UCLPowellOptimizerType;
  typedef UCLPowellOptimizerType*                                          UCLPowellOptimizerPointer;
  
  /** Registration Methods. */
  typedef itk::MaskedImageRegistrationMethod<TInputImageType>                  SingleResRegistrationType;
  typedef itk::TranslationThenRotationImageRegistrationMethod<TInputImageType> TranslationThenRotationRegistrationType;
  typedef itk::TranslateRotateScaleImageRegistrationMethod<TInputImageType>    TranslateRotateScaleRegistrationType;
  typedef itk::RigidPlusScaleImageRegistrationMethod<TInputImageType>          RigidPlusScaleRegistrationType;
  typedef itk::BlockMatchingMethod<TInputImageType, TScalarType>               BlockMatchingRegistrationType;

  /** Multi-resolution methods. */
  typedef itk::MultiResolutionImageRegistrationWrapper
                                                      <TInputImageType> MultiResRegistrationType;
  
  /** Creates a single-resolution method. */
  virtual typename SingleResRegistrationType::Pointer CreateSingleResRegistration(SingleResRegistrationMethodTypeEnum type);

  /** Creates a multi-resolution method. */
  virtual typename MultiResRegistrationType::Pointer CreateMultiResRegistration(MultiResRegistrationMethodTypeEnum type);

  /** Create an interpolator. */
  virtual typename InterpolatorType::Pointer CreateInterpolator(InterpolationTypeEnum type);

  /** Create a Metric. */
  virtual typename MetricType::Pointer CreateMetric(MetricTypeEnum type);
        
  /** Create a transform. */
  virtual typename TransformType::Pointer CreateTransform(TransformTypeEnum type);
  
  /** Create a transform from a file */
  virtual typename TransformType::Pointer CreateTransform(std::string transfomFilename); 

  /** Create an optimiser. */
  virtual typename OptimizerType::Pointer CreateOptimizer(OptimizerTypeEnum optimizerType);
  
  /** You need to create one of these, dependent on the type of optimizer. */
  virtual typename IterationUpdateCommandType::Pointer CreateIterationUpdateCommand(OptimizerTypeEnum optimizerType);
  
protected:

  ImageRegistrationFactory();
  virtual ~ImageRegistrationFactory() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  ImageRegistrationFactory(const Self&); // purposefully not implemented
  void operator=(const Self&);           // purposefully not implemented
        
};
  
} // end namespace


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageRegistrationFactory.txx"
#endif

#endif
