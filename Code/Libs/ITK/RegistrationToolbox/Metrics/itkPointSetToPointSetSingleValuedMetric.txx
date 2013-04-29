/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkPointSetToPointSetSingleValuedMetric_txx
#define _itkPointSetToPointSetSingleValuedMetric_txx

#include "itkPointSetToPointSetSingleValuedMetric.h"

namespace itk
{

/** Constructor */
template <class TFixedPointSet, class TMovingPointSet> 
PointSetToPointSetSingleValuedMetric<TFixedPointSet,TMovingPointSet>
::PointSetToPointSetSingleValuedMetric()
{
  m_FixedPointSet = 0; // has to be provided by the user.
  m_MovingPointSet   = 0; // has to be provided by the user.
  m_Transform     = 0; // has to be provided by the user.
}

/** Set the parameters that define a unique transform */
template <class TFixedPointSet, class TMovingPointSet> 
void
PointSetToPointSetSingleValuedMetric<TFixedPointSet,TMovingPointSet>
::SetTransformParameters( const ParametersType & parameters ) const
{
  if( !m_Transform )
    {
    itkExceptionMacro(<<"Transform has not been assigned");
    }
  m_Transform->SetParameters( parameters );
}


/** Initialize the metric */
template <class TFixedPointSet, class TMovingPointSet> 
void
PointSetToPointSetSingleValuedMetric<TFixedPointSet,TMovingPointSet>
::Initialize(void) throw ( ExceptionObject )
{

  if( !m_Transform )
    {
    itkExceptionMacro(<<"Transform is not present");
    }

  if( !m_MovingPointSet )
    {
    itkExceptionMacro(<<"MovingPointSet is not present");
    }

  if( !m_FixedPointSet )
    {
    itkExceptionMacro(<<"FixedPointSet is not present");
    }

  // If the PointSet is provided by a source, update the source.
  if( m_MovingPointSet->GetSource() )
    {
    m_MovingPointSet->GetSource()->Update();
    }

  // If the point set is provided by a source, update the source.
  if( m_FixedPointSet->GetSource() )
    {
    m_FixedPointSet->GetSource()->Update();
    }
}
 

/** PrintSelf */
template <class TFixedPointSet, class TMovingPointSet> 
void
PointSetToPointSetSingleValuedMetric<TFixedPointSet,TMovingPointSet>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Moving PointSet: " << m_MovingPointSet.GetPointer()  << std::endl;
  os << indent << "Fixed  PointSet: " << m_FixedPointSet.GetPointer()   << std::endl;
  os << indent << "Transform:    " << m_Transform.GetPointer()    << std::endl;
}


} // end namespace itk

#endif

