/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkUCLSimplexOptimizer_h
#define __itkUCLSimplexOptimizer_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkAmoebaOptimizer.h>

namespace itk
{
  
/** 
 * \class UCLSimplexOptimizer
 * \brief Subclass itkAmoebaOptimizer to fix bug in SetCostFunction.
 *
 * \ingroup Numerics Optimizers
 */
class NIFTKITK_WINEXPORT ITK_EXPORT UCLSimplexOptimizer : public AmoebaOptimizer
{
public:
  /** Standard "Self" typedef. */
  typedef UCLSimplexOptimizer                 Self;
  typedef AmoebaOptimizer                     Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro( UCLSimplexOptimizer, AmoebaOptimizer );

  /** Plug in a Cost Function into the optimizer  */
  virtual void SetCostFunction(SingleValuedCostFunction * costFunction );

protected:
  UCLSimplexOptimizer();
  virtual ~UCLSimplexOptimizer();

private:
  UCLSimplexOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#endif
