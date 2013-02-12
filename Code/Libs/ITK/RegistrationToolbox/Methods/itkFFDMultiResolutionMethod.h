/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFFDMultiResolutionMethod_h
#define __itkFFDMultiResolutionMethod_h

#include "itkMultiResolutionDeformableImageRegistrationMethod.h"
#include "itkBSplineTransform.h"

namespace itk
{

/** 
 * \class FFDMultiResolutionMethod
 * \brief Extends MultiResolutionDeformableImageRegistrationMethod to sort out interpolating
 * the BSpline grid in between resolution levels.
 * 
 * \sa MultiResolutionDeformableImageRegistrationMethod
 */
template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
class ITK_EXPORT FFDMultiResolutionMethod : 
public MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar> 
{
public:
  /** Standard class typedefs. */
  typedef FFDMultiResolutionMethod                                             Self;
  typedef MultiResolutionDeformableImageRegistrationMethod<TInputImageType, 
                                                           TScalarType, 
                                                           NDimensions,
                                                           TDeformationScalar> Superclass;
  typedef SmartPointer<Self>                                                   Pointer;
  typedef SmartPointer<const Self>                                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(FFDMultiResolutionMethod, MultiResolutionDeformableImageRegistrationMethod);

  /**  Type of the input image. */
  typedef TInputImageType                                                      InputImageType;
  typedef typename InputImageType::SpacingType                                 InputImageSpacingType;

  /**  Type of the Transform . */
  typedef BSplineTransform<TInputImageType, TScalarType, 
                           NDimensions, TDeformationScalar >                   BSplineTransformType;
  typedef typename BSplineTransformType::Pointer                               BSplineTransformPointer;

  typedef LocalSimilarityMeasureGradientDescentOptimizer<TInputImageType, 
                                                         TInputImageType,
                                                         TScalarType,
                                                         TDeformationScalar>   OptimizerType;
  typedef OptimizerType*                                                       OptimizerPointer;
  
  /** Set/Get the Transfrom. */
  itkSetObjectMacro( Transform, BSplineTransformType );
  itkGetObjectMacro( Transform, BSplineTransformType );

  /**
   * The max step size = max voxel size * MaxStepSizeFactor
   * 
   * and indicates the maximum step size taken per iteration at a given control point.
   * Default is 1.0, so the initial step size will equal the maximum voxel dimension.
   * Then as the registration progresses, this is reduced accordingly.
   */
  itkSetMacro(MaxStepSizeFactor, TScalarType);
  itkGetMacro(MaxStepSizeFactor, TScalarType);

  /** 
   * The min step size =  max step size * MinStepSizeFactor
   * 
   * and indicates the minimum step size taken per iteration at a given control point.
   * Defaults to 0.01, so the initial minimum step size is 0.01 times the max step size.
   * As registration progresses, the step size is reduced accordingly, and once it is below
   * this calculated threshold, registration will stop.
   */
  itkSetMacro(MinStepSizeFactor, TScalarType);
  itkGetMacro(MinStepSizeFactor, TScalarType);

  /** Write out the current transformation as an image of vectors. */
  void WriteControlPointImage(std::string filename);

  /** Sets the final control point spacing. */
  itkSetMacro(FinalControlPointSpacing, InputImageSpacingType);
  itkGetMacro(FinalControlPointSpacing, InputImageSpacingType);
  
protected:
  FFDMultiResolutionMethod();
  virtual ~FFDMultiResolutionMethod() {};

  /**
   * In here, we sort out initialising the BSpline grid at first resolution
   * level, and then interpolating the BSpline grid inbetween resolution levels.
   */
  virtual void BeforeSingleResolutionRegistration();
  
private:
  FFDMultiResolutionMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  BSplineTransformPointer m_Transform;
  
  /** to calculate minimum step size. */
  TScalarType m_MinStepSizeFactor;
  
  /** to calculate a max step size. */
  TScalarType m_MaxStepSizeFactor;
  
  /** This sets the minimum size of the control point grid. */
  InputImageSpacingType m_FinalControlPointSpacing;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFDMultiResolutionMethod.txx"
#endif

#endif



