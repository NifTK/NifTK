/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
