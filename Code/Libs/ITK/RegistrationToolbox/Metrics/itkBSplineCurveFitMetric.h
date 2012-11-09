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

#ifndef __itkBSplineCurveFitMetric_h
#define __itkBSplineCurveFitMetric_h

#include <vector>

#include "itkConceptChecking.h"
#include "itkSingleValuedCostFunction.h"
#include "itkPoint.h"


namespace itk
{
  
/** \class BSplineCurveFitMetric
 * \brief Class to compute the goodness of fit of a BSpline to a set of data values.
 */

template < class IntensityType = int >
class ITK_EXPORT BSplineCurveFitMetric : public SingleValuedCostFunction
{
  public:

  /** Standard class typedefs. */
  typedef BSplineCurveFitMetric             Self;
  typedef SingleValuedCostFunction     Superclass;
  typedef SmartPointer<Self>           Pointer;
  typedef SmartPointer<const Self>     ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineCurveFitMetric, SingleValuedCostFunction);
  
  /**  Type of the parameters. */
  typedef typename SingleValuedCostFunction::ParametersType ParametersType;
  typedef typename SingleValuedCostFunction::MeasureType    MeasureType;
  typedef typename SingleValuedCostFunction::DerivativeType DerivativeType;
  
  /** Some convenient typedefs. */

  /** Initialise the regularly spaced B-Spline control points */
  void Initialise( void );

  /** Return the number of parameters required by the Transform */
  unsigned int GetNumberOfParameters(void) const;
  
  /** This method returns the value and derivative of the cost function corresponding
    * to the specified parameters    */ 
  virtual void GetValueAndDerivative( const ParametersType & parameters,
                                      MeasureType & value,
                                      DerivativeType & derivative ) const;

  /** This method returns the value of the cost function corresponding
    * to the specified parameters. */ 
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** This method returns the derivative of the cost function corresponding
    * to the specified parameters. */ 
  virtual void GetDerivative( const ParametersType & parameters,
                              DerivativeType & derivative ) const;


protected:

  BSplineCurveFitMetric();
  virtual ~BSplineCurveFitMetric() {};

  void PrintSelf(std::ostream& os, Indent indent) const;


private:

  BSplineCurveFitMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented


};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineCurveFitMetric.txx"
#endif

#endif
