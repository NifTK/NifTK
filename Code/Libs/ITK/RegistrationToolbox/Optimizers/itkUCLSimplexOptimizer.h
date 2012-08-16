/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkUCLSimplexOptimizer_h
#define __itkUCLSimplexOptimizer_h

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include "itkAmoebaOptimizer.h"

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
