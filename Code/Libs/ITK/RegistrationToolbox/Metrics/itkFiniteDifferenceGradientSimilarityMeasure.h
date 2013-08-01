/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkFiniteDifferenceGradientSimilarityMeasure_h
#define itkFiniteDifferenceGradientSimilarityMeasure_h

#include "itkSimilarityMeasure.h"


namespace itk
{
/** 
 * \class FiniteDifferenceGradientSimilarityMeasure
 * \brief AbstractBase class, just to implement the finite difference gradient method.
 *
 * Note that this class is NOT thread safe.
 * 
 * \ingroup RegistrationMetrics
 */
template < typename TFixedImage, typename TMovingImage > 
class ITK_EXPORT FiniteDifferenceGradientSimilarityMeasure : 
    public SimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef FiniteDifferenceGradientSimilarityMeasure      Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >  Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(FiniteDifferenceGradientSimilarityMeasure, SimilarityMeasure);

  /** The scales type. */
  typedef Array<double> ScalesType;

  /** Types transferred from the base class */
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::MeasureType                MeasureType;
  
  /** Get the derivatives of the match measure. */
  virtual void GetCostFunctionDerivative( const TransformParametersType & parameters, DerivativeType  & derivative ) const;

  /**
   * Define the transform and thereby the parameter space of the metric
   * and the space of its derivatives.
   */
  void SetTransform( TransformType * transform );

  /** Sets the step length used to calculate the derivative. Default 0.1, which will be terrible for scale parameters. */
  itkSetMacro( DerivativeStepLength, double );

  /** Returns the step length used to calculate the derivative. Default 0.1, which will be terrible for scale parameters. */
  itkGetMacro( DerivativeStepLength, double );

  /** Sets the derivative step length scales. Default 1. ie. all parameters equally scaled. */
  itkSetMacro( DerivativeStepLengthScales, ScalesType );

  /** Returns the derivate step length scales. Default 1. ie. all parameters equally scaled. */
  itkGetConstReferenceMacro(DerivativeStepLengthScales, ScalesType);

  /** 
   * Depending on registration type, we may or may not need the scale array.
   * Here we set a flag to decide if we resize it to the same size as the
   * parameter array. For example, if we are doing NMI using finite differences,
   * we need the scales array. If however, we are calculating NMI, but are
   * calculating gradient at control points (like Free Form), or at
   * voxels (like Fluid), then we dont need this array, and additionally
   * for deformable registration, this array could be huge, so we
   * make sure we dont have to incur the memory cost. Defaults to true.
   */
  itkSetMacro(UseDerivativeScaleArray, bool);
  itkGetMacro(UseDerivativeScaleArray, bool);
  
protected:
  
  FiniteDifferenceGradientSimilarityMeasure();
  virtual ~FiniteDifferenceGradientSimilarityMeasure() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  
  FiniteDifferenceGradientSimilarityMeasure(const Self&); // purposefully not implemented
  void operator=(const Self&);                            // purposefully not implemented

  /** Flag to decide if we are using derivative scale array. Default to false. */
  bool m_UseDerivativeScaleArray;
  
  /** The step length used to calculate the derivative. */
  double m_DerivativeStepLength;

  /** The derivative step length scales. */
  ScalesType m_DerivativeStepLengthScales;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFiniteDifferenceGradientSimilarityMeasure.txx"
#endif

#endif



