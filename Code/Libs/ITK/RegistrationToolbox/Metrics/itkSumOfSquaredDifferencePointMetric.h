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
#ifndef __itkSumOfSquaredDifferencePointMetric_h
#define __itkSumOfSquaredDifferencePointMetric_h

#include "itkPointSetToPointSetSingleValuedMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"

namespace itk
{
/** 
 * \class SumOfSquaredDifferencePointMetric
 * \brief Computes sum of the squared distance between two point sets.
 * 
 * This measure was implemented as part of section 2.2 in 
 * Ourselin et. al. Image And Vision Computing 19 (2000) 25-31, for our
 * NifTK block matching implementation. We have a standard 
 * sum of squared difference (this class), and also a sum of manhattan distance
 * between two points sets.
 * 
 * \sa AbsoluteManhattanDistancePointMetric
 * \sa 
 * \ingroup RegistrationMetrics
 */
template < class TFixedPointSet, class TMovingPointSet>
class ITK_EXPORT SumOfSquaredDifferencePointMetric : 
    public PointSetToPointSetSingleValuedMetric< TFixedPointSet, TMovingPointSet>
{
public:

  /** Standard class typedefs. */
  typedef SumOfSquaredDifferencePointMetric                                       Self;
  typedef PointSetToPointSetSingleValuedMetric<TFixedPointSet, TMovingPointSet >  Superclass;
  typedef SmartPointer<Self>                                                      Pointer;
  typedef SmartPointer<const Self>                                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(SumOfSquaredDifferencePointMetric, PointSetToPointSetSingleValuedMetric);
 
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
  SumOfSquaredDifferencePointMetric();
  virtual ~SumOfSquaredDifferencePointMetric() {};

  /** PrintSelf funtion */
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  SumOfSquaredDifferencePointMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSumOfSquaredDifferencePointMetric.txx"
#endif

#endif
