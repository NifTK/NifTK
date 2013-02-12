/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKLOCALSIMILARITYMEASUREGRADIENTDESCENTOPTIMIZER_H_
#define ITKLOCALSIMILARITYMEASUREGRADIENTDESCENTOPTIMIZER_H_

#include "NifTKConfigure.h"
#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkDeformableTransform.h"
#include "itkSimilarityMeasure.h"

namespace itk
{
	
/** 
 * \class LocalSimilarityMeasureGradientDescentOptimizer
 * \brief Implement a gradient descent optimization suitable for FFD and Fluid deformation.
 *
 * \ingroup Numerics Optimizers
 */  
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
class ITK_EXPORT LocalSimilarityMeasureGradientDescentOptimizer :
    public SingleValuedNonLinearOptimizer
{
public:
  
  /** 
   * Standard class typedefs. 
   */
  typedef LocalSimilarityMeasureGradientDescentOptimizer  Self;
  typedef SingleValuedNonLinearOptimizer                  Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Standard Type Macro. */
  itkTypeMacro( LocalSimilarityMeasureGradientDescentOptimizer, SingleValuedNonLinearOptimizer );

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs. */
  typedef SimilarityMeasure<TFixedImage, TMovingImage>         ImageToImageMetricType;
  typedef const ImageToImageMetricType*                        ImageToImageMetricPointer;
  typedef TFixedImage                                          FixedImageType;
  typedef typename FixedImageType::PixelType                   FixedImagePixelType;
  typedef FixedImageType*                                      FixedImagePointer;
  typedef typename FixedImageType::Pointer                     FixedImageSmartPointer;
  typedef TMovingImage                                         MovingImageType;
  typedef typename MovingImageType::PixelType                  MovingImagePixelType;  
  typedef MovingImageType*                                     MovingImagePointer;
  typedef DeformableTransform<TFixedImage, TScalarType, Dimension, TDeformationScalar>  DeformableTransformType;
  typedef typename DeformableTransformType::Pointer            DeformableTransformPointer;
  typedef double                                               MeasureType;
  typedef InterpolateImageFunction< TFixedImage, TScalarType > RegriddingInterpolatorType;
  typedef typename RegriddingInterpolatorType::Pointer         RegriddingInterpolatorPointer;
  typedef ResampleImageFilter< TFixedImage, TFixedImage >      ResampleFilterType;
  typedef typename ResampleFilterType::Pointer                 ResampleFilterPointer;
  typedef typename DeformableTransformType::JacobianDeterminantFilterType::OutputImageType JacobianImageType; 

  /** Start optimization. */
  virtual void StartOptimization( void );

  /** Resume previously stopped optimization with current parameters
   * \sa StopOptimization. */
  virtual void ResumeOptimization( void );

  /** Stop optimization.
   * \sa ResumeOptimization */
  virtual void StopOptimization( void );

  /** Set the maximize flag. */
  itkGetConstReferenceMacro( Maximize, bool );
  itkSetMacro( Maximize, bool );
  itkBooleanMacro( Maximize );

  /** 
   * Get the current value of the cost function. 
   */
  itkGetConstMacro( Value, double );

  /** 
   * Get the value of the stop flag.
   */
  itkGetConstMacro( Stop, bool );

  /** Set/Get the maximum number of iterations. */
  itkSetMacro( MaximumNumberOfIterations, unsigned long int );
  itkGetMacro( MaximumNumberOfIterations, unsigned long int );
  
  /** Get the current iteration number. */
  itkGetMacro( CurrentIteration, unsigned long int );
  
  /** Set/Get the initial step size. */
  itkSetMacro ( StepSize, double );
  itkGetMacro ( StepSize, double );

  /** Set/Get the minimum step size. */
  itkSetMacro ( MinimumStepSize, double );
  itkGetMacro ( MinimumStepSize, double );

  /** If we can't find a better value of the transformation, we reduce step size and keep hunting. */
  itkSetMacro ( IteratingStepSizeReductionFactor, double );
  itkGetMacro ( IteratingStepSizeReductionFactor, double );

  /** If we decide to regrid, we can also tweak the step size and keep hunting. */
  itkSetMacro ( RegriddingStepSizeReductionFactor, double );
  itkGetMacro ( RegriddingStepSizeReductionFactor, double );

  /** If for some reason, the Jacobian is negative, we reduce the step size and keep hunting. */
  itkSetMacro ( JacobianBelowZeroStepSizeReductionFactor, double );
  itkGetMacro ( JacobianBelowZeroStepSizeReductionFactor, double );

  /** If the largest magnitude deformation vector is < MinimumDeformationMagnitudeThreshold, we stop. */
  itkSetMacro ( MinimumDeformationMagnitudeThreshold, double );
  itkGetMacro ( MinimumDeformationMagnitudeThreshold, double );

  /** If any voxel Jacobian drops below MinimumJacobianThreshold we regrid. */
  itkSetMacro ( MinimumJacobianThreshold, double );
  itkGetMacro ( MinimumJacobianThreshold, double );

  /** Set the regridding interpolator. */
  itkSetObjectMacro(RegriddingInterpolator, RegriddingInterpolatorType);
  itkGetConstObjectMacro(RegriddingInterpolator,  RegriddingInterpolatorType );

  /** Set the deformable transform externally. */
  itkSetObjectMacro(DeformableTransform, DeformableTransformType);
  itkGetConstObjectMacro(DeformableTransform,  DeformableTransformType );

  /** So we can force optimizer to stop if cost function not doing much. */
  itkSetMacro ( MinimumSimilarityChangeThreshold, double );
  itkGetMacro ( MinimumSimilarityChangeThreshold, double );
  
  /**
   * Get/Set similarity measure checking flag. 
   */
  itkSetMacro ( CheckSimilarityMeasure, bool );
  itkGetMacro ( CheckSimilarityMeasure, bool );

  /**
   * Get/Set MinimumDeformationMagnitudeThreshold flag. 
   */
  itkSetMacro ( CheckMinDeformationMagnitudeThreshold, bool );
  itkGetMacro ( CheckMinDeformationMagnitudeThreshold, bool );

  /**
   * Get/Set the CheckJacobianBelowZero flag.
   */
  itkSetMacro ( CheckJacobianBelowZero, bool );
  itkGetMacro ( CheckJacobianBelowZero, bool );
  
  /** At each iteration, flag to decide if we dump the next set of parameters. */
  itkSetMacro ( WriteNextParameters, bool );
  itkGetMacro ( WriteNextParameters, bool );
  
  /** At each iteration, filename to decide where we dump the next set of parameters. Default "tmp.next.params". */
  itkSetMacro ( NextParametersFileName, std::string );
  itkGetMacro ( NextParametersFileName, std::string );

  /** At each iteration, filename to decide where we dump the next set of parameters. Default "vtk". */
  itkSetMacro ( NextParametersFileExt, std::string );
  itkGetMacro ( NextParametersFileExt, std::string );

  /** At each iteration, flag to decide if we dump the deformation field. */
  itkSetMacro ( WriteDeformationField, bool );
  itkGetMacro ( WriteDeformationField, bool );
  
  /** At each iteration, filename to decide where we dump the deformation field. Default "tmp.next.field". */
  itkSetMacro ( DeformationFieldFileName, std::string );
  itkGetMacro ( DeformationFieldFileName, std::string );

  /** At each iteration, filename to decide where we dump the next set of parameters. Default "vtk". */
  itkSetMacro ( DeformationFieldFileExt, std::string );
  itkGetMacro ( DeformationFieldFileExt, std::string );

  /** Each time we regrid, flag to decide if we dump the regridded image. */
  itkSetMacro ( WriteRegriddedImage, bool );
  itkGetMacro ( WriteRegriddedImage, bool );
  
  /** Each time we regrid, filename to dump regridded image to. Default "tmp.regridded". */
  itkSetMacro ( RegriddedImageFileName, std::string );
  itkGetMacro ( RegriddedImageFileName, std::string );

  /** Each time we regrid, file extension to dump regridded image to. Default "nii". */
  itkSetMacro ( RegriddedImageFileExt, std::string );
  itkGetMacro ( RegriddedImageFileExt, std::string );

  /** Set/Get the regridded moving image pad value. */
  itkSetMacro ( RegriddedMovingImagePadValue, MovingImagePixelType );
  itkGetMacro ( RegriddedMovingImagePadValue, MovingImagePixelType );
  
  /** Set/Get IsAbsRegriddedImage */
  itkSetMacro(IsAbsRegriddedImage, bool); 
  itkGetMacro(IsAbsRegriddedImage, bool); 
  
  /**
   * Set/Get.
   */
  itkSetMacro(IsPropagateRegriddedMovingImage, bool); 
  itkGetMacro(IsPropagateRegriddedMovingImage, bool); 
  
  /**
   * Return the composed Jacobian. 
   */
  virtual const JacobianImageType* GetComposedJacobian() const 
  { 
    return this->m_ComposedJacobian; 
  }
  /**
   * Return the regridded moving image. 
   */
  virtual const FixedImageType* GetRegriddedMovingImage() const 
  {
    return this->m_RegriddedMovingImage; 
  }
  /**
   * Return the regridded fixed image. 
   */
  virtual const FixedImageType* GetRegriddedFixedImage() const 
  {
    return this->m_RegriddedFixedImage; 
  }
  
protected:
  
  LocalSimilarityMeasureGradientDescentOptimizer(); 
  virtual ~LocalSimilarityMeasureGradientDescentOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Set whether we are maximizing or not. */
  bool m_Maximize;
  
  /**
   * The current value of the cost function/similarity measure. 
   */
  double m_Value;

  /**
   * Set this flag to stop the optimisation. 
   */ 
  bool m_Stop;

  /**
   * The max. number of iterations. 
   */   
  unsigned long int m_MaximumNumberOfIterations;
  
  /**
   * The current number of iterations.
   */
  unsigned int m_CurrentIteration;

  /** The number of the times we have regridded. */
  unsigned int m_RegriddingNumber;
  
  /** Step Size. */
  double m_StepSize;
  
  /** Minimum step size. */
  double m_MinimumStepSize;
  
  /** If we can't find a better position, we reduce the step size, and keep iterating. */
  double m_IteratingStepSizeReductionFactor;

  /** If Jacobian hits our lowest threshold, and we regrid, we also change step size and keep iterating. */
  double m_RegriddingStepSizeReductionFactor;
  
  /** If for some reason, Jacobian is negative, we reduce step size, and keep iterating. */
  double m_JacobianBelowZeroStepSizeReductionFactor;
  
  /** 
   * If the deformation is < this threshold, we stop.
   */
  double m_MinimumDeformationMagnitudeThreshold;
  
  /** If the similarity doesn't improve over this threshold, we stop. */
  double m_MinimumSimilarityChangeThreshold;
  
  /**
   * If the Jacobian is < this threshold, we regrid.
   */
  double m_MinimumJacobianThreshold;
  
  /**
   * Check for better similarity measure or not. 
   */
  bool m_CheckSimilarityMeasure; 
  
  /** Check for min deformation change or not. Default true. */
  bool m_CheckMinDeformationMagnitudeThreshold;
  
  /** Check for jacobian below zero or not. Default true. */
  bool m_CheckJacobianBelowZero;
  
  /** Set to true to apply an abs filter to the regridded image. */
  bool m_IsAbsRegriddedImage; 
  
  /** For any initialisation of optimizer. */
  virtual void Initialize() {};
  
  /** Calculate a potential step following the gradient direction. */
  virtual double CalculateNextStep(int iterationNumber, double currentSimilarity, const ParametersType& current, ParametersType& next) = 0;

  /** For any cleanup of optimizer. */
  virtual void CleanUp() {};

  /** Do the re-gridding.*/
  virtual void ReGrid(bool isResetCurrentPosition);
  
  /**
   * Compose the new Jacobian using current and regridded parameters. 
   */
  virtual void ComposeJacobian()
  {                               
    typedef MultiplyImageFilter<JacobianImageType> MultiplyImageFilterType; 
    
    this->m_DeformableTransform->ComputeMinJacobian(); 
    typename MultiplyImageFilterType::Pointer multiplyFilter = MultiplyImageFilterType::New(); 
    multiplyFilter->SetInput1(this->m_ComposedJacobian); 
    multiplyFilter->SetInput2(this->m_DeformableTransform->GetJacobianImage()); 
    multiplyFilter->Update(); 
    this->m_ComposedJacobian = multiplyFilter->GetOutput(); 
    this->m_ComposedJacobian->DisconnectPipeline(); 
  }
  
  /** So we can set a regridding interpolator, different from the one we use during registration. */
  RegriddingInterpolatorPointer m_RegriddingInterpolator;
  
  /** To actually do the resampling. */
  ResampleFilterPointer m_RegriddingResampler;

  /** Regridded moving image, resampled to same type as fixed image. */
  FixedImageSmartPointer m_RegriddedMovingImage;
  
  /** Regridded moving image, resampled to same type as fixed image. */
  FixedImageSmartPointer m_RegriddedFixedImage;
  
  /** The deformable transform. */
  DeformableTransformPointer m_DeformableTransform;

  /** The potential next parameters (subject to Jacobian/Deformation/Cost function checks). */
  ParametersType m_NextParameters;
  
  /** Store the regrid */
  ParametersType m_RegriddedParameters;

  /** Metric. */
  ImageToImageMetricPointer m_ImageToImageMetric;

  /** Fixed image. */
  FixedImagePointer m_FixedImage;
  
  /** Moving image. */
  MovingImagePointer m_MovingImage;
  
  /** Writes the next parameters at each iteration. */
  bool m_WriteNextParameters;
  
  /** Filename for next parameters at each iteration. */
  std::string m_NextParametersFileName; 
  
  /** File extension for next parameters at each iteration. */
  std::string m_NextParametersFileExt;

  /** Writes the deformation field at each iteration (to check that the parameters were applied properly. */
  bool m_WriteDeformationField;

  /** File name for deformation field. */
  std::string m_DeformationFieldFileName;
  
  /** File extension for deformation field. */
  std::string m_DeformationFieldFileExt;
  
  /** So we can optionally turn on dumping out the regridded image. */
  bool m_WriteRegriddedImage;
  
  /** Filename for regridded image. */
  std::string m_RegriddedImageFileName;

  /** File extension for regridded image. */
  std::string m_RegriddedImageFileExt;
  
  /** Pad value for regridded image. Default 0. */
  MovingImagePixelType m_RegriddedMovingImagePadValue;
  
  /**
   * Composed Jacobian image calculated along the regridding.  
   */
  typename JacobianImageType::Pointer m_ComposedJacobian; 
  
  /**
   * Propagate the regridded moving image when re-gridding, if set to true. 
   * Exaxctly the same as original Christensen fluid paper. 
   * The reason for preferring this is that the interpolation of 
   * the image may be more accurate than the interpolation of thee
   * deformation field.
   */
  bool m_IsPropagateRegriddedMovingImage; 
  
private:

  LocalSimilarityMeasureGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLocalSimilarityMeasureGradientDescentOptimizer.txx"
#endif

#endif /*ITKLOCALSIMILARITYMEASUREGRADIENTDESCENTOPTIMIZER_H_*/



