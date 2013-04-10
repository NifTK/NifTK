/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef __itkBSplineCurveFitMetric_txx
#define __itkBSplineCurveFitMetric_txx

#include "itkBSplineCurveFitMetric.h"

#include "itkUCLMacro.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template< class IntensityType >
BSplineCurveFitMetric< IntensityType >
::BSplineCurveFitMetric()
{

}



/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template< class IntensityType >
void
BSplineCurveFitMetric< IntensityType >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


/* -----------------------------------------------------------------------
   Initialise()
   ----------------------------------------------------------------------- */

template< class IntensityType >
void
BSplineCurveFitMetric< IntensityType >
::Initialise( void )
{

}


/* -----------------------------------------------------------------------
   GetNumberOfParameters()
   ----------------------------------------------------------------------- */

template< class IntensityType >
unsigned int 
BSplineCurveFitMetric< IntensityType >
::GetNumberOfParameters( void ) const
{
  return 0;
}


/* -----------------------------------------------------------------------
   GetNumberOfValues()
   ----------------------------------------------------------------------- */

template< class IntensityType >
unsigned int 
BSplineCurveFitMetric< IntensityType >
::GetNumberOfValues( void ) const
{
  return 0;
}


/* -----------------------------------------------------------------------
   GetValue() - Get the value of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType >
typename BSplineCurveFitMetric< IntensityType >::MeasureType
BSplineCurveFitMetric< IntensityType >
::GetValue( const ParametersType &parameters ) const
{
  niftkitkDebugMacro("BSplineCurveFitMetric::GetValue()");

  MeasureType currentMeasure;

  currentMeasure.Fill( 0. );

  std::cout << "Current cost: " << currentMeasure << std::endl;

  return currentMeasure;
}


/* -----------------------------------------------------------------------
   GetDerivative() - Get the derivative of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType >
void
BSplineCurveFitMetric< IntensityType >
::GetDerivative( const ParametersType &parameters, 
                 DerivativeType &derivative ) const
{
  niftkitkDebugMacro("BSplineCurveFitMetric::GetDerivative()");

}


/* -----------------------------------------------------------------------
   GetValueAndDerivative() - Get both the value and derivative of the metric
   ----------------------------------------------------------------------- */

template< class IntensityType >
void
BSplineCurveFitMetric< IntensityType >
::GetValueAndDerivative(const ParametersType &parameters, 
                        MeasureType &Value, DerivativeType &Derivative) const
{
  niftkitkDebugMacro("BSplineCurveFitMetric::GetValueAndDerivative()");

  // Compute the similarity

  Value = this->GetValue( parameters );

  // Compute the derivative
  
  this->GetDerivative( parameters, Derivative );
}

} // end namespace itk


#endif
