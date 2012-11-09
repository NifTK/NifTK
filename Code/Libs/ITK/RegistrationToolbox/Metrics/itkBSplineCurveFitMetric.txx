/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-10-31 16:15:38 +0000 (Wed, 31 Oct 2012) $
 Revision          : $Revision: 9614 $
 Last modified by  : $Author: jhh $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
   GetValue() - Get the value of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType >
typename BSplineCurveFitMetric< IntensityType >::MeasureType
BSplineCurveFitMetric< IntensityType >
::GetValue( const ParametersType &parameters ) const
{
  niftkitkDebugMacro("BSplineCurveFitMetric::GetValue()");

  MeasureType currentMeasure = 0;

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
