/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkAbsoluteManhattanDistancePointMetric_h
#define __itkAbsoluteManhattanDistancePointMetric_h

#include "itkPointSetToPointSetSingleValuedMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"

namespace itk
{
/** 
 * \class AbsoluteManhattanDistancePointMetric
 * \brief Computes the sum of the absolute Manhattan Distance (L1-norm) between two point sets.
 * 
 * This measure was implemented as part of section 2.2 in 
 * Ourselin et. al. Image And Vision Computing 19 (2000) 25-31.
 * The aim is to register two point sets, using a robust convex M-estimator.
 * The paper suggests that this is simply the absolute value of your distance
 * measure. The paper also suggests manhattan distance is better than Euclidean
 * distance. Hence this class. It takes two point sets, of exactly the same
 * number of points, in corresponding order, and computes the sum
 * of the absolute manhattan distance between corresponding points.
 * Manhattan distance is also known as an L1-norm, but I like Manhattan.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedPointSet, class TMovingPointSet>
class ITK_EXPORT AbsoluteManhattanDistancePointMetric : 
    public PointSetToPointSetSingleValuedMetric< TFixedPointSet, TMovingPointSet>
{
public:

  /** Standard class typedefs. */
  typedef AbsoluteManhattanDistancePointMetric                                    Self;
  typedef PointSetToPointSetSingleValuedMetric<TFixedPointSet, TMovingPointSet >  Superclass;
  typedef SmartPointer<Self>                                                      Pointer;
  typedef SmartPointer<const Self>                                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(AbsoluteManhattanDistancePointMetric, PointSetToPointSetSingleValuedMetric);
 
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

protected:
  AbsoluteManhattanDistancePointMetric();
  virtual ~AbsoluteManhattanDistancePointMetric() {};

  /** PrintSelf funtion */
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  AbsoluteManhattanDistancePointMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAbsoluteManhattanDistancePointMetric.txx"
#endif

#endif
