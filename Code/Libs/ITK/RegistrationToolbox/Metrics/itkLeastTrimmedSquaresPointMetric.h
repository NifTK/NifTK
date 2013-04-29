/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkLeastTrimmedSquaresPointMetric_h
#define __itkLeastTrimmedSquaresPointMetric_h

#include "itkPointSetToPointSetSingleValuedMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"

namespace itk
{
/** 
 * \class LeastTrimmedSquaresPointMetric
 * \brief Like sum of squared difference between point sets, except you throw away
 * a certain percentage of outliers.
 * 
 * This measure was implemented in section 2.2 equation 1 of 
 * Ourselin et. al. MICCAI 2000, "Block Matching: a general framework..."
 * Given two corresponding point sets, we compute the sum of squared difference
 * of the smallest h% percentage. This will throw away gross outliers.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedPointSet, class TMovingPointSet>
class ITK_EXPORT LeastTrimmedSquaresPointMetric : 
    public PointSetToPointSetSingleValuedMetric< TFixedPointSet, TMovingPointSet>
{
public:

  /** Standard class typedefs. */
  typedef LeastTrimmedSquaresPointMetric                                          Self;
  typedef PointSetToPointSetSingleValuedMetric<TFixedPointSet, TMovingPointSet >  Superclass;
  typedef SmartPointer<Self>                                                      Pointer;
  typedef SmartPointer<const Self>                                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(LeastTrimmedSquaresPointMetric, PointSetToPointSetSingleValuedMetric);
 
  /** Types transferred from the base class */
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;

  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::FixedPointSetType          FixedPointSetType;
  typedef typename Superclass::MovingPointSetType         MovingPointSetType;
  typedef typename Superclass::FixedPointSetConstPointer  FixedPointSetConstPointer;
  typedef typename Superclass::MovingPointSetConstPointer MovingPointSetConstPointer;

  typedef typename Superclass::PointIterator              PointIterator;
  typedef typename Superclass::PointDataIterator          PointDataIterator;

  /** Get the number of values */
  unsigned int GetNumberOfValues() const;

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters, DerivativeType & Derivative ) const;

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters, MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set the percentage to keep when we evaluate the sum. Defaults to 50%*/
  itkSetMacro(PercentageOfPointsToKeep, int);
  itkGetMacro(PercentageOfPointsToKeep, int);
  
protected:
  LeastTrimmedSquaresPointMetric();
  virtual ~LeastTrimmedSquaresPointMetric() {};

  /** PrintSelf funtion */
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  LeastTrimmedSquaresPointMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  int m_PercentageOfPointsToKeep;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLeastTrimmedSquaresPointMetric.txx"
#endif

#endif
