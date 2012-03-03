/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-15 14:34:51 +0000 (Thu, 15 Dec 2011) $
 Revision          : $Revision: 8026 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKFLUIDGRADIENTDESCENTOPTIMIZER_H_
#define ITKFLUIDGRADIENTDESCENTOPTIMIZER_H_

#include "NifTKConfigure.h"
#include "itkLocalSimilarityMeasureGradientDescentOptimizer.h"
#include "itkImageToImageMetricWithConstraint.h"
#include "itkFluidDeformableTransform.h"
#include "itkRegistrationForceFilter.h"
#include "itkFluidPDEFilter.h"
#include "itkFluidVelocityToDeformationFilter.h"
#include "itkIterationUpdateCommand.h"
#include "itkRecursiveGaussianImageFilter.h"
#include "itkDBCImageFilter.h"

namespace itk
{
  
/** 
 * \class FluidGradientDescentOptimizer
 * \brief Class to perform fluid specific optimization.
 *
 * \ingroup Numerics Optimizers
 */  
template <class TFixedImage, class TMovingImage, class TScalar, class TDeformationScalar>
class FluidGradientDescentOptimizer :
public LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage, TMovingImage, TScalar, TDeformationScalar>
{
public:
  
  /** 
   * Standard class typedefs. 
   */
  typedef FluidGradientDescentOptimizer                                              Self;
  typedef LocalSimilarityMeasureGradientDescentOptimizer<TFixedImage, TMovingImage, TScalar, TDeformationScalar>  Superclass;
  typedef SmartPointer<Self>                                                         Pointer;
  typedef SmartPointer<const Self>                                                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard Type Macro. */
  itkTypeMacro( FluidGradientDescentOptimizer, LocalSimilarityMeasureGradientDescentOptimizer );

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs. */
  typedef FluidDeformableTransform<TFixedImage, TScalar, Dimension, TDeformationScalar> DeformableTransformType;
  typedef typename DeformableTransformType::Pointer                               DeformableTransformPointer;
  typedef double                                                                  MeasureType;
  typedef ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>             SimilarityMeasureType;
  typedef typename SimilarityMeasureType::ConstPointer                            SimilarityMeasurePointer;
  typedef typename SimilarityMeasureType::TransformParametersType                 ParametersType;
  typedef FluidDeformableTransform<TFixedImage, TScalar, 
                                   Dimension, TDeformationScalar>                 FluidTransformType;
  typedef FluidTransformType*                                                     FluidTransformPointer;
  
  typedef Vector< double, itkGetStaticConstMacro(Dimension) >                     VectorPixelType;
  typedef Image< VectorPixelType, itkGetStaticConstMacro(Dimension) >             VectorImageType;

  /** Fluid Pipeline as follows: First generate the force. */
  typedef RegistrationForceFilter<TFixedImage, TMovingImage, TDeformationScalar> ForceFilterType;
  typedef typename ForceFilterType::Pointer                                      ForceFilterPointer;
  
  /** Fluid Pipeline as follows: Then solve the PDE */
  typedef typename itk::FluidPDEFilter<TDeformationScalar, Dimension>            FluidPDEType;
  typedef typename FluidPDEType::Pointer                                         FluidPDEPointer;

  /** Fluid Pipeline as follows: Then add the velocity to current deformation */
  typedef typename itk::FluidVelocityToDeformationFilter<TDeformationScalar, Dimension >   FluidAddVelocityFilterType;
  typedef typename FluidAddVelocityFilterType::Pointer                      FluidAddVelocityFilterPointer;
  typedef typename FluidAddVelocityFilterType::OutputImageType              OutputImageType;
  typedef typename OutputImageType::PixelType                               OutputImagePixelType;
  typedef typename OutputImageType::Pointer                                 OutputImagePointer;
  typedef ImageRegionIterator<OutputImageType>                              OutputImageIteratorType;
  typedef typename OutputImageType::SizeType                                OutputImageSizeType;
  typedef Image<float, Dimension>                                           MaskType; 
  typedef ImageRegionIteratorWithIndex<MaskType> AsgdMaskIteratorType; 
  typedef typename Superclass::JacobianImageType JacobianImageType; 
 
  /** DBC */
  typedef Image<short, itkGetStaticConstMacro(Dimension)>  DBCMaskType; 
  typedef DBCImageFilter<TFixedImage, DBCMaskType>         DBCFilterType; 
  typedef ResampleImageFilter<DBCMaskType, DBCMaskType>    DBCMaskResampleFilterType; 
  typedef NearestNeighborInterpolateImageFunction<DBCMaskType, double> DBCNearestInterpolatorType;   // opt for speed. 

  /** Set the force filter to use. */ 
  itkSetObjectMacro( ForceFilter, ForceFilterType );
  itkGetConstObjectMacro( ForceFilter, ForceFilterType );

  /** Set the PDE Solver */
  itkSetObjectMacro(FluidPDESolver, FluidPDEType);
  itkGetConstObjectMacro(FluidPDESolver,  FluidPDEType );

  /** Set the filter that adds the velocity to the current field. */
  itkSetObjectMacro(FluidVelocityToDeformationFilter, FluidAddVelocityFilterType);
  itkGetConstObjectMacro(FluidVelocityToDeformationFilter,  FluidAddVelocityFilterType );
  
  /**
   * Set/Get. 
   */
  itkSetObjectMacro(FixedImageTransform, FluidTransformType); 
  itkGetObjectMacro(FixedImageTransform, FluidTransformType); 
  itkSetObjectMacro(MovingImageInverseTransform, FluidTransformType); 
  itkGetObjectMacro(MovingImageInverseTransform, FluidTransformType); 
  itkSetObjectMacro(FixedImageInverseTransform, FluidTransformType); 
  itkGetObjectMacro(FixedImageInverseTransform, FluidTransformType); 
  itkSetObjectMacro(FluidVelocityToFixedImageDeformationFilter, FluidAddVelocityFilterType); 
  itkGetObjectMacro(FluidVelocityToFixedImageDeformationFilter, FluidAddVelocityFilterType); 
  itkSetObjectMacro(DBCFilter, DBCFilterType); 
  itkSetObjectMacro(FixedImageDBCMask, DBCMaskType); 
  itkSetObjectMacro(MovingImageDBCMask, DBCMaskType); 

  /**
   * Set/Get macro. 
   */
  itkSetMacro(MinimumDeformationMaximumIterations, int); 
  itkGetMacro(MinimumDeformationMaximumIterations, int); 
  itkSetMacro(MinimumDeformationAllowedForIterations, double); 
  itkGetMacro(MinimumDeformationAllowedForIterations, double); 
  itkSetMacro(IsSymmetric, bool); 
  itkGetMacro(IsSymmetric, bool); 
  itkSetMacro(DBCStepSizeTrigger, double); 
  itkSetMacro(UseJacobianInForce, bool); 
  
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
  typename DeformableTransformType::DeformableParameterPointerType GetCurrentDeformableParameters() { return this->m_CurrentDeformableParameters; } 
  
  /**
   * Set adaptive size gradient descent parameters.  
   */
  void SetAsgdParameter(double asgdA, double asgdFMax, double asgdFMin, double asgdW, double asgdFMinFudgeFactor, double asgdWFudgeFactor) 
  {
    this->m_AsgdA = asgdA; 
    this->m_AsgdFMax = asgdFMax;   
    this->m_AsgdFMin = asgdFMin;   
    this->m_AsgdW = asgdW;   
    this->m_AsgdWFudgeFactor = asgdWFudgeFactor; 
    this->m_AsgdFMinFudgeFactor = asgdFMinFudgeFactor; 
  }
  
  /**
   * Set adaptive size gradient descent mask. 
   */
  void SetAsgdMask(typename MaskType::Pointer mask)
  {
    this->m_AsgdMask = mask; 
  }
  
  /**
   * Return the composed Jacobian. 
   */
  virtual const JacobianImageType* GetMovingImageTransformComposedJacobianForward() const 
  { 
    return this->m_MovingImageTransformComposedJacobianForward; 
  }
  /**
   * Return the composed Jacobian. 
   */
  virtual const JacobianImageType* GetMovingImageTransformComposedJacobianBackward() const 
  { 
    return this->m_MovingImageTransformComposedJacobianBackward; 
  }
  /**
   * Return the composed Jacobian. 
   */
  virtual const JacobianImageType* GetFixedImageTransformComposedJacobianForward() const 
  { 
    return this->m_FixedImageTransformComposedJacobianForward; 
  }
  /**
   * Return the composed Jacobian. 
   */
  virtual const JacobianImageType* GetFixedImageTransformComposedJacobianBackward() const 
  { 
    return this->m_FixedImageTransformComposedJacobianBackward; 
  }
  
protected:
  
  FluidGradientDescentOptimizer(); 
  virtual ~FluidGradientDescentOptimizer() {};
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
   * Compute the best step size using golden section search. 
   */
  virtual double ComputeBestStepSize(typename DeformableTransformType::DeformableParameterPointerType current, typename DeformableTransformType::DeformableParameterPointerType next, typename DeformableTransformType::DeformableParameterPointerType currentFixed, typename DeformableTransformType::DeformableParameterPointerType nextFixed, double currentSimilarity, double* bestSimilarity); 
  /**
   * Compute the current similarity measure given the step size. 
   */
  virtual double ComputeSimilarityMeasure(typename DeformableTransformType::DeformableParameterPointerType current, typename DeformableTransformType::DeformableParameterPointerType next, typename DeformableTransformType::DeformableParameterPointerType currentFixed, typename DeformableTransformType::DeformableParameterPointerType nextFoward, double stepSize); 
  
  /**
   * Regridding. 
   */
  virtual void ReGrid(bool isResetCurrentPosition); 
  
  /**
   * Return the fluid tranform type point. 
   */
  DeformableTransformType* GetFluidDeformableTransform() 
  { 
    FluidTransformPointer transform = dynamic_cast<FluidTransformPointer>(this->m_DeformableTransform.GetPointer());
    if (transform == 0)
    {
      itkExceptionMacro(<< "Can't dynamic cast to FluidTransformPointer");
    }
    return transform;
  }
  
  /**
   * Compose the new Jacobian using current and regridded parameters. 
   */
  virtual void ComposeJacobian(); 
      
  /**
   * Estimate the variance of the gradient error by randomly sampling the images. 
   */
  double EstimateGradientErrorVariance(int numberOfSamples, int numberOfSimulations, typename FluidPDEType::OutputImageType::Pointer originalVelocity, double* eta); 
  
  /**
   * Estimate step sizes using adaptive size gradient descent
   */
  void EstimateSimpleAdapativeStepSize(double* bestStepSize, double* bestFixedImageStepSize); 
  
  /**
  * Estimate step sizes using adaptive size gradient descent, 
  * see Frassoldati et al (2008) Journal of industrial and management optimization. 
  */
  void EstimateAdapativeBarzilaiBorweinStepSize(double* bestStepSize, double* bestFixedImageStepSize); 
  
  /**
   * Perform DBC. 
   */
  void PerformDBC(bool isCalculateBiasField); 

  /** We inject a force filter. */
  ForceFilterPointer m_ForceFilter;
  
  /** To solve the Fluid PDE. */
  FluidPDEPointer m_FluidPDESolver;
  
  /** To convert velocity to the field. */
  FluidAddVelocityFilterPointer m_FluidVelocityToDeformationFilter;
  
  /** To convert velocity to the field. */
  FluidAddVelocityFilterPointer m_FluidVelocityToFixedImageDeformationFilter;
  
  /**
   * Normalise step size to the one specified by the user. 
   */
  bool m_NormaliseStepSize; 
  
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
   * Calcualte the velocity field if set to true. 
   */
  bool m_CalculateVelocityFeild; 
  /** 
   * Current deformation field 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_CurrentDeformableParameters; 
  /** 
   * The potential next parameters (subject to Jacobian/Deformation/Cost function checks). 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_NextDeformableParameters; 
  /** 
   * Store the regrid parameters. 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_RegriddedDeformableParameters; 
  /**
   * Initial deformable paremters (e.g. from another level). 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_InitialDeformableParameters; 
  /**
   * Fixed image transform. 
   */
  typename FluidTransformType::Pointer m_FixedImageTransform; 
  /** 
   * Current deformation field 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_CurrentFixedDeformableParameters; 
  /** 
   * The potential next parameters (subject to Jacobian/Deformation/Cost function checks). 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_NextFixedDeformableParameters; 
  /** 
   * Store the regrid parameters. 
   */
  typename DeformableTransformType::DeformableParameterPointerType m_RegriddedFixedDeformableParameters; 
  /**
   * Symmetric?
   */
  bool m_IsSymmetric; 
  /**
   * Time step for the adaptive gradient descent. 
   */
  double m_AdjustedTimeStep; 
  /**
   * Time step for the adaptive gradient descent. 
   */
  double m_AdjustedFixedImageTimeStep; 
  /**
   * Previous gradient. 
   */
  typename FluidAddVelocityFilterType::OutputImageType::Pointer m_PreviousGradient; 
  /**
   * Previous gradient for fixed image transform. 
   */
  typename FluidAddVelocityFilterType::OutputImageType::Pointer m_PreviousFixedImageGradient; 
  /** 
   * To actually do the resampling. 
   */
  typename Superclass::ResampleFilterPointer m_RegriddingFixedImageResampler;
  /**
   * Adaptive size gradient descent: F_max.
   */
  double m_AsgdFMax; 
  /**
   * Adaptive size gradient descent: F_min.
   */
  double m_AsgdFMin; 
  /**
   * Adaptive size gradient descent: w.
   */
  double m_AsgdW; 
  /**
   * adaptive size gradient descent: mask. 
   */
  typename MaskType::Pointer m_AsgdMask; 
  /**
   * Current velocity field. 
   */
  typename FluidPDEType::OutputImageType::Pointer m_CurrentVelocityField; 
  /**
   * Fudge factor for converting the variance to w. 
   */
  double m_AsgdWFudgeFactor; 
  /**
   * Fudge factor for converting the error 
   */
  double m_AsgdFMinFudgeFactor; 
  /**
   * Adaptive size gradient descent: A.
   */
  double m_AsgdA; 
  /**
   * Step size history. 
   */
  std::vector<double> m_StepSizeHistoryForMovingImage; 
  /**
   * Step size history. 
   */
  std::vector<double> m_StepSizeHistoryForFixedImage; 
  /**
   * Composed Jacobian image of the moving image transform calculated along the regridding 
   * in the forward direction.  
   */
  typename JacobianImageType::Pointer m_MovingImageTransformComposedJacobianForward; 
  /**
   * Composed Jacobian image of the moving image transform calculated along the regridding 
   * in the backward direction.  
   */
  typename JacobianImageType::Pointer m_MovingImageTransformComposedJacobianBackward; 
  /**
   * Composed Jacobian image of the fixed image transform calculated along the regridding 
   * in the forward direction.  
   */
  typename JacobianImageType::Pointer m_FixedImageTransformComposedJacobianForward; 
  /**
   * Composed Jacobian image of the fixed image transform calculated along the regridding 
   * in the backward direction.  
   */
  typename JacobianImageType::Pointer m_FixedImageTransformComposedJacobianBackward; 
  /**
   * Moving image inverse transform. 
   */
  typename DeformableTransformType::Pointer m_MovingImageInverseTransform; 
  /**
   * Fixed image inverse transform. 
   */
  typename DeformableTransformType::Pointer m_FixedImageInverseTransform; 
  /**
   * DBC filter. 
   */
  typename DBCFilterType::Pointer m_DBCFilter; 
  /**
   * Fixed image DBC mask. 
   */
  typename DBCMaskType::Pointer m_FixedImageDBCMask; 
  /**
   * Moving image DBC mask. 
   */
  typename DBCMaskType::Pointer m_MovingImageDBCMask; 
  /**
   * Currently accumulated step size. 
   */
  double m_DBCStepSize; 
  /**
   * Step size trigger for re-calculating bias fields. . 
   */
  double m_DBCStepSizeTrigger; 
  /**
   * Use Jacobian in calculating force. 
   */
  bool m_UseJacobianInForce; 
  
  
private:

  FluidGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFluidGradientDescentOptimizer.txx"
#endif

#endif /*ITKFLUIDGRADIENTDESCENTOPTIMIZER_H_*/



