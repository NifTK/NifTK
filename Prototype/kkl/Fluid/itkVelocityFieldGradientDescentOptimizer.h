/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-27 13:54:26 +0100 (Fri, 27 May 2011) $
 Revision          : $Revision: 6300 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKVelocityFieldGradientDescentOptimizer_H_
#define ITKVelocityFieldGradientDescentOptimizer_H_


#include "itkFluidGradientDescentOptimizer.h"
#include "itkImageToImageMetricWithConstraint.h"
#include "itkVelocityFieldDeformableTransform.h"
#include "itkRegistrationForceFilter.h"
#include "itkFluidPDEFilter.h"
#include "itkFluidVelocityToDeformationFilter.h"
#include "itkIterationUpdateCommand.h"
#include "itkRecursiveGaussianImageFilter.h"

namespace itk
{
  
/** 
 * \class VelocityFieldGradientDescentOptimizer
 * \brief Class to perform fluid specific optimization.
 *
 * \ingroup Numerics Optimizers
 */  
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
class VelocityFieldGradientDescentOptimizer : 
public FluidGradientDescentOptimizer<TFixedImage, TMovingImage, TScalar, TDeformationScalar>
{
public:
  /** 
   * Standard class typedefs. 
   */
  typedef VelocityFieldGradientDescentOptimizer Self;
  typedef FluidGradientDescentOptimizer<TFixedImage, TMovingImage, TScalar, TDeformationScalar>  Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard Type Macro. */
  itkTypeMacro( VelocityFieldGradientDescentOptimizer, FluidGradientDescentOptimizer );

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs. */
  typedef VelocityFieldDeformableTransform<TFixedImage, TScalar, Dimension, TDeformationScalar> DeformableTransformType;
  typedef typename DeformableTransformType::Pointer                               DeformableTransformPointer;
  typedef double                                                                  MeasureType;
  typedef ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>             SimilarityMeasureType;
  typedef typename SimilarityMeasureType::TransformParametersType                 ParametersType;
  typedef VelocityFieldDeformableTransform<TFixedImage, TScalar, Dimension, TDeformationScalar> VelocityFieldTransformType;
  typedef Vector< double, itkGetStaticConstMacro(Dimension) >                     VectorPixelType;
  typedef Image< VectorPixelType, itkGetStaticConstMacro(Dimension) >             VectorImageType;
  typedef typename DeformableTransformType::DeformableParameterType DeformableParameterType; 
  typedef InterpolateImageFunction< TFixedImage, TScalar > InterpolatorType;
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, TDeformationScalar> ForceFilterType;
  typedef FluidPDEFilter<TDeformationScalar, Dimension> FluidPDEType;
  typedef typename Superclass::ImageToImageMetricType ImageToImageMetricType; 
  typedef Image<float, TFixedImage::ImageDimension> StepSizeImageType; 

  /**
   * Set/Get macro. 
   */
  itkSetMacro(MinimumDeformationMaximumIterations, int); 
  itkGetMacro(MinimumDeformationMaximumIterations, int); 
  itkSetMacro(MinimumDeformationAllowedForIterations, double); 
  itkGetMacro(MinimumDeformationAllowedForIterations, double); 
  itkSetObjectMacro( ForceFilter, ForceFilterType );
  itkGetConstObjectMacro( ForceFilter, ForceFilterType );
  itkSetObjectMacro(FluidPDESolver, FluidPDEType);
  itkGetConstObjectMacro(FluidPDESolver,  FluidPDEType );
  itkSetObjectMacro(FixedImageInterpolator, InterpolatorType); 
  itkGetObjectMacro(FixedImageInterpolator, InterpolatorType); 
  itkSetObjectMacro(MovingImageInterpolator, InterpolatorType); 
  itkGetObjectMacro(MovingImageInterpolator, InterpolatorType); 
  
  /** Start optimization. */
  virtual void StartOptimization( void );
  
  /** 
   * Resume previously stopped optimization with current parameters
   * \sa StopOptimization. 
  */
  virtual void ResumeOptimization(void);
  
  /**
   * Set/Get initial deformable parameters. 
   */
  typename DeformableTransformType::DeformableParameterPointerType GetInitialDeformableParameters() { return this->m_InitialDeformableParameters; }
  void SetInitialDeformableParameters(typename DeformableTransformType::DeformableParameterPointerType parameters) { this->m_InitialDeformableParameters = parameters; }
  
  /**
   * Get the current deformable parameters. 
   */
  typename DeformableTransformType::DeformableParameterPointerType GetCurrentMovingImageDeformableParameters() { return this->m_CurrentDeformableParameters; } 
  
  /**
   * Save step size image. 
   */
  static void SaveStepSizeImage(typename StepSizeImageType::Pointer field, std::string filename); 
  
  /**
   * Load step size image. 
   */
  static void LoadStepSizeImage(typename StepSizeImageType::Pointer& field, std::string filename); 
  
protected:
  
  VelocityFieldGradientDescentOptimizer(); 
  virtual ~VelocityFieldGradientDescentOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** 
   * Calculate a potential step following the gradient direction. 
   */
  virtual double CalculateNextStep(int iterationNumber, double currentSimilarity, const ParametersType& current, ParametersType& next)
  {
    itkExceptionMacro(<< "This should not be called. It uses too much the memory.");
  }
  
  /** 
   * Calculate a potential step following the gradient direction. 
   */
  virtual double CalculateNextStep(int iterationNumber, double currentSimilarity, typename DeformableTransformType::DeformableParameterPointerType current, typename DeformableTransformType::DeformableParameterPointerType & next, typename DeformableTransformType::DeformableParameterPointerType currentFixed, typename DeformableTransformType::DeformableParameterPointerType & nextFixed);
  
  /**
   * Return the velocity field tranform type point. 
   */
  DeformableTransformType* GetMovingImageVelocityFieldDeformableTransform() 
  { 
    DeformableTransformType* transform = dynamic_cast<DeformableTransformType*>(this->m_DeformableTransform.GetPointer());
    if (transform == 0)
    {
      itkExceptionMacro(<< "Can't dynamic cast to FluidTransformPointer");
    }
    return transform;
  }
  
  /**
   * Return the velocity field tranform type point. 
   */
  DeformableTransformType* GetFixedImageVelocityFieldDeformableTransform() 
  { 
    DeformableTransformType* transform = dynamic_cast<DeformableTransformType*>(this->m_FixedImageTransform.GetPointer());
    if (transform == 0)
    {
      itkExceptionMacro(<< "Can't dynamic cast to FluidTransformPointer");
    }
    return transform;
  }

  /** We inject a force filter. */
  typename ForceFilterType::Pointer m_ForceFilter;
  
  /** To solve the Fluid PDE. */
  typename FluidPDEType::Pointer m_FluidPDESolver;
  
  /**
   * Remember the starting step size after regridding. 
   */
  double m_StartingStepSize; 
  
  /**
   * Current iterations with step size less than the minimum deformation. 
   */
  int m_CurrentMinimumDeformationIterations; 
  
  /**
   * User specified max iteration allowed for step size less than the minimum deformation. 
   */
  int m_MinimumDeformationMaximumIterations; 
  
  /**
   * User specified min deformation. 
   */
  double m_MinimumDeformationAllowedForIterations; 
  
  /**
   * Resample the fixed image during registration. 
   */
  typename InterpolatorType::Pointer m_FixedImageInterpolator; 
  
  /**
   * Resample the moving image during registration. 
   */
  typename InterpolatorType::Pointer m_MovingImageInterpolator; 
  
  double m_InitialStepSize; 
  
  /**
   * Previous velcoity gradients. 
   */
  typename std::vector<typename DeformableTransformType::DeformableParameterType::Pointer>  m_PreviousVelocityFieldGradient; 
  
  /**
   * Step size images. 
   */
  typename std::vector<typename StepSizeImageType::Pointer> m_StepSizeImage; 

  /**
   * Step size images. 
   */
  typename std::vector<typename StepSizeImageType::Pointer> m_StepSizeNormalisationFactorImage; 
  
  /**
   * Iteration with the best similarity. 
   */
  unsigned int m_BestIteration; 
  
  /**
   * The number of iteration with worse similarity allowed. 
   */
  unsigned int m_WorseNumberOfIterationsAllowed; 
  
  /**
   * Normalisation factor for step size. 
   */
  double m_NormalisationFactor; 
  
private:

  VelocityFieldGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVelocityFieldGradientDescentOptimizer.txx"
#endif

#endif /*ITKVelocityFieldGradientDescentOptimizer_H_*/



