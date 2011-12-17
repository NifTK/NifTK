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
#ifndef _itkAbsoluteManhattanDistancePointMetric_txx
#define _itkAbsoluteManhattanDistancePointMetric_txx

#include "itkAbsoluteManhattanDistancePointMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/** Constructor */
template <class TFixedPointSet, class TMovingPointSet> 
AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::AbsoluteManhattanDistancePointMetric() 
{
}

/** Return the number of values, i.e the number of points in the moving set */
template <class TFixedPointSet, class TMovingPointSet>  
unsigned int
AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>  
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
typename AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>::MeasureType
AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>
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

      // Compute L1 norm, a.k.a. Manhattan distance.
      // as mentioned on page 28, line 9.
      distance = 0;
      for (i = 0; i < TFixedPointSet::PointDimension; i++)
        {
          distance += fabs(movingPoint[i] - transformedPoint[i]);	
        }

      // Compute equation 3, using p(x) = |x| as mentioned
      // on page 27, line 9 from the bottom.
    
      measure += fabs(distance);

      ++fixedPointItr;
      ++movingPointItr;
    }
  return measure;
}

/** Get the Derivative Measure */
template <class TFixedPointSet, class TMovingPointSet>
void
AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetDerivative( const TransformParametersType & itkNotUsed(parameters),
                 DerivativeType & itkNotUsed(derivative) ) const
{

}

/** Get both the match Measure and theDerivative Measure  */
template <class TFixedPointSet, class TMovingPointSet>  
void
AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::GetValueAndDerivative(const TransformParametersType & parameters, 
                        MeasureType & value, DerivativeType  & derivative) const
{
  value = this->GetValue(parameters);
  this->GetDerivative(parameters,derivative);
}

/** PrintSelf method */
template <class TFixedPointSet, class TMovingPointSet>  
void
AbsoluteManhattanDistancePointMetric<TFixedPointSet,TMovingPointSet>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk


#endif
