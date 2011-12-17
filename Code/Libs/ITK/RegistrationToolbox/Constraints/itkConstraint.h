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
#ifndef __itkConstraint_h
#define __itkConstraint_h

#include "itkObject.h"
#include "itkObjectFactory.h"

#include "itkSingleValuedCostFunction.h"

namespace itk
{
  
/** 
 * \class Constraint
 * \brief Abstract Base class for constraints, which are objects that return 
 * a single double value, such as might be used for a regulariser in a
 * deformable registration algorithm. In practice you create any subclass
 * of this.  Then the base similarity measure itkImageToImageMetricWithConstraint
 * will evaluate the cost function, for example, the mutual information of 
 * your two images is X, then, if the itkImageToImageMetricWithConstraint
 * has a pointer to a constraint, it will call EvaluateConstraint, which 
 * will return a value Y, then itkImageToImageMetricWithConstraint will
 * combine those two numbers together. So, in principal, the 
 * itkImageToImageMetricWithConstraint (and derived subclasses) need not
 * even know what or how a constraint works, only that it provides a number.
 * 
 * \ingroup RegistrationMetrics
 */
class ITK_EXPORT Constraint : 
    public Object
{
public:
  /** Standard "Self" typedef. */
  typedef Constraint                               Self;
  typedef Object                                   Superclass;
  typedef SmartPointer<Self>                       Pointer;
  typedef SmartPointer<const Self>                 ConstPointer;
  typedef SingleValuedCostFunction::MeasureType    MeasureType;
  typedef SingleValuedCostFunction::DerivativeType DerivativeType;
  typedef SingleValuedCostFunction::ParametersType ParametersType;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( Constraint, Object );

  /** Derived classes must simply return a number. */
  virtual MeasureType EvaluateConstraint(const ParametersType & parameters) = 0;
  
  /** Derived class must write the derivative of the constraint into the supplied derivative array. */
  virtual void EvaluateDerivative(const ParametersType & parameters, DerivativeType & derivative ) const = 0;
  
protected:
  
  Constraint() {};
  virtual ~Constraint() {};

private:  
  
  Constraint(const Self&);          // purposely not implemented
  void operator=(const Self&);      // purposely not implemented
};

} // end namespace itk

#endif
