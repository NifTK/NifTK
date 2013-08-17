/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkImageToImageMetricWithConstraint_h
#define itkImageToImageMetricWithConstraint_h

#include <itkImageToImageMetric.h>
#include <itkConstraint.h>
#include "itkMetricDerivativeBridge.h"

namespace itk
{
/** 
 * \class ImageToImageMetricWithConstraint
 * \brief Abstract base class to provide functionality for adding arbitrary
 * constraints, and also arbitrary ways of evaluating a derivative,
 * both via Template Method pattern [2].
 * 
 * We also provide a boolean to turn the gradient of the constraint off/on,
 * as this could be expensive, and also, an independent debugging parameter.
 * When you evaluate the similarity measure, do you want to print out the result
 * for debugging purposes? i.e. like:
 * 
 * "GetValue():Actual metric value x = similarity * (1-weighting) + constraintg * (weighting)"
 * 
 * This may be ok for Fluid,FFD or normal affine registration where you evaluate the similarity
 * measure once per iteration, or once + NDOF*2 if you include the finite difference derivative.
 * But its completely useless for block matching, where you may evaluate the similarity measure
 * (albeit on a small block) many thousands of times per iteration.
 * So, we can't turn this feature off/on based on logging, or else you would need different
 * logging every time you ran a block matching. We can't turn this feature off/on using
 * the base class, or global debug flag.  So, ive added a boolean m_PrintOutMetricEvaluation,
 * which defaults to true.
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT ImageToImageMetricWithConstraint : 
    public ImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef ImageToImageMetricWithConstraint                  Self;
  typedef ImageToImageMetric<TFixedImage, TMovingImage >    Superclass;
  typedef SmartPointer<Self>                                Pointer;
  typedef SmartPointer<const Self>                          ConstPointer;

  /** This class enables a global constraint to be added. */
  typedef itk::Constraint                                   ConstraintType;
  typedef typename ConstraintType::Pointer                  ConstraintPointer;

  /** This class enables a derivative bridge to be added. */
  typedef MetricDerivativeBridge<TFixedImage, TMovingImage> MetricDerivativeBridgeType;
  typedef typename MetricDerivativeBridgeType::Pointer      MetricDerivativePointer;
  
  /** For parameters and derivatives. */
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::ParametersType               TransformParametersType;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageToImageMetricWithConstraint, ImageToImageMetric);

  /** Methods to Set/Get an optional derivative bridge. */
  itkSetObjectMacro(DerivativeBridge, MetricDerivativeBridgeType);
  itkGetObjectMacro(DerivativeBridge, MetricDerivativeBridgeType);

  /** Set/Get an optional constraint. */
  itkSetObjectMacro( Constraint, ConstraintType );
  itkGetObjectMacro( Constraint, ConstraintType );

  /** Set the weighting factor. */
  itkSetMacro( WeightingFactor, double );
  itkGetMacro( WeightingFactor, double );

  /** Turn the derivative of constraint on or off, as this could be very expensive. */
  itkSetMacro( UseConstraintGradient, bool );
  itkGetConstMacro( UseConstraintGradient, bool );

  /** Turn on/off the printing out of the metric evaluation. Default on. */
  itkSetMacro( PrintOutMetricEvaluation, bool );
  itkGetMacro( PrintOutMetricEvaluation, bool );

  /**
   * Get the value of the cost function, which will include the weighted constraint. */
  virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** 
   * Takes the supplied parameters array, and a derivative array, calculate the 
   * derivative of the constraint, and ADDS it to the supplied array.
   */
  virtual void GetConstraintDerivative( const TransformParametersType & parameters, DerivativeType  & derivative ) const;
  
  /** 
   * Get the derivatives of the cost function, which can include the 
   * derivative of the constraint if UseConstraintGradient is true. 
   * 
   * Note that this method is provided here in this base class, so that potentially
   * any registration method could use this mechanism.  However, if you look
   * at the itkFFDGradientDescentOptimizer, you will notice that the BSpline stuff doesnt
   * call this GetDerivative. i.e. the registration is not driven by a regular optimizer
   * that calls GetValueAndDerivative.
   */
  virtual void GetDerivative( const TransformParametersType & parameters, DerivativeType  & derivative ) const;

  /**  Simply calls GetValue and then GetDerivative, both of which could be virtual/overriden. */
  virtual void GetValueAndDerivative( const TransformParametersType & parameters, MeasureType& Value, DerivativeType& derivative ) const;

protected:
  
  ImageToImageMetricWithConstraint();
  virtual ~ImageToImageMetricWithConstraint() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Derived classes must implement this to calculate the similarity measure. */
  virtual MeasureType GetSimilarity( const TransformParametersType & parameters ) const = 0;
  
  /** Derived classes must implement this to calculate the derivative of the cost function. */
  virtual void GetCostFunctionDerivative( const TransformParametersType & parameters, DerivativeType  & derivative ) const = 0;

  /**
   * A pointer to a MetricDerivativeBrige which can be used to 
   * delegate to potentially anything that can evaluate the derivative of the cost function.
   */
  MetricDerivativePointer m_DerivativeBridge;
  
  /** 
   * A pointer to a constraint. 
   * This can be anything that returns a simple number.
   */
  ConstraintPointer m_Constraint;

  /** 
   * The weighting between the constraint and the cost function.
   * Default 0.01 (99% cost function).
   */
  double m_WeightingFactor;

  /** Turn derivative of constraint on or off. Default off. */
  bool m_UseConstraintGradient;
  
  /** Print out the metric evaluation. Default true. */
  bool m_PrintOutMetricEvaluation;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToImageMetricWithConstraint.txx"
#endif

#endif



