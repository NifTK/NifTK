/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkKnownCorEuclideanDistancePointMetric_txx
#define __itkKnownCorEuclideanDistancePointMetric_txx

#include "itkKnownCorEuclideanDistancePointMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/** Constructor */
template <class TFixedPointSet, class TMovingPointSet> 
KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::KnownCorEuclideanDistancePointMetric()
{
  // when set to true it will be a bit faster, but it will result in minimizing
  // the sum of distances^4 instead of the sum of distances^2
  m_ComputeSquaredDistance = false; 
}

/** Return the number of values, i.e the number of points in the moving set */
template <class TFixedPointSet, class TMovingPointSet>  
unsigned int
KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>  
::GetNumberOfValues() const
{
  MovingPointSetConstPointer movingPointSet = this->GetMovingPointSet();

  if( !movingPointSet ) 
    {
    itkExceptionMacro( << "Moving point set has not been assigned" );
    }

  return  movingPointSet->GetPoints()->Size();
}


/** Get the match Measure */
template <class TFixedPointSet, class TMovingPointSet>  
typename KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>::MeasureType
KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetValue( const TransformParametersType & parameters ) const
{
  double sum = 0.;
  FixedPointSetConstPointer fixedPointSet = this->GetFixedPointSet();

  if( !fixedPointSet ) 
    {
    itkExceptionMacro( << "Fixed point set has not been assigned" );
    }

  
  MovingPointSetConstPointer movingPointSet = this->GetMovingPointSet();

  if( !movingPointSet ) 
    {
    itkExceptionMacro( << "Moving point set has not been assigned" );
    }

  PointIterator pointItr = movingPointSet->GetPoints()->Begin();
  PointIterator pointEnd = movingPointSet->GetPoints()->End();
  // --> second iterator to run through the fixed set of points 
  PointIterator pointItr2 = fixedPointSet->GetPoints()->Begin();

  MeasureType measure;
  measure.set_size(movingPointSet->GetPoints()->Size());

  this->SetTransformParameters( parameters );

  unsigned int identifier = 0;
  while( pointItr != pointEnd )
    {
    typename Superclass::InputPointType  inputPoint;
    inputPoint.CastFrom( pointItr.Value() );
    typename Superclass::OutputPointType transformedPoint = 
      this->m_Transform->TransformPoint( inputPoint );

    double dist = pointItr2.Value().SquaredEuclideanDistanceTo(transformedPoint);    

#if 0
    std::cout << "For point set: " << identifier << " the fixedPixel is " << pointItr2.Value() 
	      << " and the movingPixel is: " << pointItr.Value() << " Their distance is: " << dist << std::endl;
#endif

    if(!m_ComputeSquaredDistance)
      {
      dist = vcl_sqrt(dist);
      }

    sum += dist;

    measure.put(identifier, dist);

    ++pointItr;
    ++pointItr2;
    identifier++;
    }

#if 0
  std::cout << "Measure: " << measure << std::endl;
#endif
  std::cout << "Sum of distances: " << sum << std::endl;
  
  return measure;
}

/** Get the Derivative Measure */
template <class TFixedPointSet, class TMovingPointSet>
void
KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetDerivative( const TransformParametersType & itkNotUsed(parameters),
                 DerivativeType & itkNotUsed(derivative) ) const
{

}

/** Get both the match Measure and theDerivative Measure  */
template <class TFixedPointSet, class TMovingPointSet>  
void
KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetValueAndDerivative(const TransformParametersType & parameters, 
                        MeasureType & value, DerivativeType  & derivative) const
{
  value = this->GetValue(parameters);
  this->GetDerivative(parameters,derivative);
}

/** PrintSelf method */
template <class TFixedPointSet, class TMovingPointSet>  
void
KnownCorEuclideanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if(m_ComputeSquaredDistance)
    {
    os << indent << "m_ComputeSquaredDistance: True"<< std::endl;
    }
  else
    {
    os << indent << "m_ComputeSquaredDistance: False"<< std::endl;
    }
}

} // end namespace itk


#endif
