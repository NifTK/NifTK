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

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkConjugateGradientMaxIterOptimizer_h
#define __itkConjugateGradientMaxIterOptimizer_h

#include "itkSingleValuedNonLinearVnlOptimizer.h"
#include <vnl/algo/vnl_conjugate_gradient.h>

namespace itk
{

/** \class ConjugateGradientMaxIterOptimizer
 * \brief Wrap of the vnl_conjugate_gradient
 *
 * \ingroup Numerics Optimizers
 */
class ITK_EXPORT ConjugateGradientMaxIterOptimizer :
    public SingleValuedNonLinearVnlOptimizer

{
public:
  /** Standard class typedefs. */
  typedef ConjugateGradientMaxIterOptimizer          Self;
  typedef SingleValuedNonLinearVnlOptimizer   Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( ConjugateGradientMaxIterOptimizer, SingleValuedNonLinearOptimizer );

  /** InternalParameters typedef. */
  typedef   vnl_vector<double>     InternalParametersType;

  /** Internal Optimizer Type */
  typedef   vnl_conjugate_gradient InternalOptimizerType;

  /** Method for getting access to the internal optimizer */
  vnl_conjugate_gradient * GetOptimizer(void);

  /** Set/Get the maximum number of function evaluations allowed. */
  virtual void SetMaximumNumberOfFunctionEvaluations( unsigned int n );
  itkGetMacro( MaximumNumberOfFunctionEvaluations, unsigned int );

  /** Start optimization with an initial value. */
  void StartOptimization( void );

  /** Plug in a Cost Function into the optimizer  */
  virtual void SetCostFunction( SingleValuedCostFunction * costFunction );


  /** Return the number of iterations performed so far */
  unsigned long GetNumberOfIterations(void) const;
  unsigned long GetCurrentIteration(void) const;

  /** Return Current Value */
  MeasureType GetValue() const;

protected:
  ConjugateGradientMaxIterOptimizer();
  virtual ~ConjugateGradientMaxIterOptimizer();

  typedef Superclass::CostFunctionAdaptorType   CostFunctionAdaptorType;

private:
  ConjugateGradientMaxIterOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /**  The vnl optimization method for conjugate gradient. */
  bool                         		m_OptimizerInitialized;
  InternalOptimizerType      	* m_VnlOptimizer;
  unsigned int                		 m_MaximumNumberOfFunctionEvaluations;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkConjugateGradientMaxIterOptimizer.txx"
#endif

#endif
