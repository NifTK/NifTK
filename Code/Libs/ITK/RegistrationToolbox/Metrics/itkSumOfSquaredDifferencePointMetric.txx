/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkSumOfSquaredDifferencePointMetric_txx
#define _itkSumOfSquaredDifferencePointMetric_txx

#include "itkSumOfSquaredDifferencePointMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/** Constructor */
template <class TFixedPointSet, class TMovingPointSet> 
SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>
::SumOfSquaredDifferencePointMetric() 
{
}

/** Return the number of values, i.e the number of points in the moving set */
template <class TFixedPointSet, class TMovingPointSet>  
unsigned int
SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>  
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
typename SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>::MeasureType
SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>
::GetValue( const TransformParametersType & parameters ) const
{
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

  this->SetTransformParameters( parameters );

  PointIterator fixedPointItr = fixedPointSet->GetPoints()->Begin();
  PointIterator fixedPointEnd = fixedPointSet->GetPoints()->End();
  PointIterator movingPointItr = movingPointSet->GetPoints()->Begin();
  PointIterator movingPointEnd = movingPointSet->GetPoints()->End();
  
  MeasureType measure = 0;
  double diff;
  double distance;
  unsigned int i;

  while( fixedPointItr != fixedPointEnd && movingPointItr != movingPointEnd)
    {
      typename Superclass::InputPointType fixedPoint;
      fixedPoint.CastFrom( fixedPointItr.Value() );

      typename Superclass::InputPointType movingPoint;
      movingPoint.CastFrom( movingPointItr.Value() );
    
      typename Superclass::OutputPointType transformedPoint = 
        this->m_Transform->TransformPoint( fixedPoint );

      distance = 0;

      for (i = 0; i < TFixedPointSet::PointDimension; i++)
        {
          diff = movingPoint[i] - transformedPoint[i];
          distance += diff*diff;
        }

      measure += distance;

      ++fixedPointItr;
      ++movingPointItr;
    }
  return measure;
}

/** Get the Derivative Measure */
template <class TFixedPointSet, class TMovingPointSet>
void
SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>
::GetDerivative( const TransformParametersType & itkNotUsed(parameters),
                 DerivativeType & itkNotUsed(derivative) ) const
{

}

/** Get both the match Measure and theDerivative Measure  */
template <class TFixedPointSet, class TMovingPointSet>  
void
SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>
::GetValueAndDerivative(const TransformParametersType & parameters, 
                        MeasureType & value, DerivativeType  & derivative) const
{
  value = this->GetValue(parameters);
  this->GetDerivative(parameters,derivative);
}

/** PrintSelf method */
template <class TFixedPointSet, class TMovingPointSet>  
void
SumOfSquaredDifferencePointMetric<TFixedPointSet,TMovingPointSet>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk


#endif
