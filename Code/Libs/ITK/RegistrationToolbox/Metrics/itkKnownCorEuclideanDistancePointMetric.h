/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkKnownCorEuclideanDistancePointMetric_h
#define itkKnownCorEuclideanDistancePointMetric_h

#include <itkPointSetToPointSetMetric.h>
#include <itkCovariantVector.h>
#include <itkPoint.h>
#include <itkPointSet.h>
#include <itkImage.h>

namespace itk
{
/** \class EuclideanDistancePointMetric
 * \brief Computes the distance between a moving point-set
 *  and a fixed point-set. Correspondance is assumed between
 * consecutive elements of the two point sets.
 *
 *  Reference: "A Method for Registration of 3-D Shapes",
 *             IEEE PAMI, Vol 14, No. 2, February 1992
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedPointSet, class TMovingPointSet >
class ITK_EXPORT KnownCorEuclideanDistancePointMetric : 
    public PointSetToPointSetMetric< TFixedPointSet, TMovingPointSet>
{
public:

  /** Standard class typedefs. */
  typedef KnownCorEuclideanDistancePointMetric                                Self;
  typedef PointSetToPointSetMetric<TFixedPointSet, TMovingPointSet >  Superclass;

  typedef SmartPointer<Self>         Pointer;
  typedef SmartPointer<const Self>   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(EuclideanDistancePointMetric, Object);
 
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
  void GetDerivative( const TransformParametersType & parameters,
                      DerivativeType & Derivative ) const;

  /**  Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
                              MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set/Get if the distance should be squared. Default is true for computation speed */
  itkSetMacro(ComputeSquaredDistance,bool);
  itkGetConstMacro(ComputeSquaredDistance,bool);
  itkBooleanMacro(ComputeSquaredDistance);

protected:
  KnownCorEuclideanDistancePointMetric();
  virtual ~KnownCorEuclideanDistancePointMetric() {};

  /** PrintSelf funtion */
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  KnownCorEuclideanDistancePointMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool               m_ComputeSquaredDistance;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkKnownCorEuclideanDistancePointMetric.txx"
#endif

#endif
