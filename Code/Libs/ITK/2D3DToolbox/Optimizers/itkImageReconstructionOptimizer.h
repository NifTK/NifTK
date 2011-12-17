/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkImageReconstructionOptimizer_h
#define __itkImageReconstructionOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"


namespace itk
{
  
/** 
 * \class ImageReconstructionOptimizer
 * \brief Base class for image reconstruction optimization methods
 *
 * \ingroup Numerics Optimizers
 */

class ITK_EXPORT ImageReconstructionOptimizer : 
  public SingleValuedNonLinearOptimizer
{
public:
  /** Standard "Self" typedef. */
  typedef ImageReconstructionOptimizer   Self;
  typedef SingleValuedNonLinearOptimizer           Superclass;
  typedef SmartPointer<Self>                       Pointer;
  typedef SmartPointer<const Self>                 ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageReconstructionOptimizer, 
                SingleValuedNonLinearOptimizer );

  /** Start optimization. */
  void    StartOptimization( void );

  /** Resume previously stopped optimization with current parameters.
   * \sa StopOptimization */
  void    ResumeOptimization( void );

  /** Stop optimization.
   * \sa ResumeOptimization */
  void    StopOptimization( void );

  /** Set/Get parameters to control the optimization process. */
  itkSetMacro( NumberOfIterations, unsigned long );
  itkGetConstReferenceMacro( NumberOfIterations, unsigned long );
  itkGetConstMacro( CurrentIteration, unsigned int );

  itkGetConstReferenceMacro( StopCondition, StopConditionType );
  itkGetConstReferenceMacro( Value, MeasureType );
  
protected:
  ImageReconstructionOptimizer();
  virtual ~ImageReconstructionOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:  
  ImageReconstructionOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&);//purposely not implemented

protected:

  bool                          m_Stop;

  MeasureType                   m_Value;

  StopConditionType             m_StopCondition;

  unsigned long                 m_NumberOfIterations;
  unsigned long                 m_CurrentIteration;

};

} // end namespace itk

#endif
