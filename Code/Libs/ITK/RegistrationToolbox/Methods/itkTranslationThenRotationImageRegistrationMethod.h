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
#ifndef __itkTranslationThenRotationImageRegistrationMethod_h
#define __itkTranslationThenRotationImageRegistrationMethod_h


#include "itkMultiStageImageRegistrationMethod.h"

namespace itk
{

/** 
 * \class TranslationThenRotationImageRegistrationMethod
 * \brief Class specifically for doing registration that alternately
 * solves the translation components, then the rotation components.
 *
 * The aim is to interleave:
 * 
 * 1. Transations only, using brain image.
 * 
 * 2. Rotations only, using brain image.
 * 
 *  
 * \ingroup RegistrationFilters
 */
template <typename TInputImageType>
class ITK_EXPORT TranslationThenRotationImageRegistrationMethod 
: public MultiStageImageRegistrationMethod<TInputImageType> 
{
public:

  /** Standard class typedefs. */
  typedef TranslationThenRotationImageRegistrationMethod               Self;
  typedef MultiStageImageRegistrationMethod<TInputImageType>           Superclass;
  typedef SmartPointer<Self>                                           Pointer;
  typedef SmartPointer<const Self>                                     ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(TranslationThenRotationImageRegistrationMethod, MultiStageImageRegistrationMethod);

  /** Typedefs. */
  typedef typename Superclass::MetricType                MetricType;
  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::TransformType             TransformType;
  typedef typename Superclass::TransformType*            TransformPointer;
  typedef typename Superclass::OptimizerType             OptimizerType;                   
  typedef typename Superclass::OptimizerType*            OptimizerPointer;
  typedef typename Superclass::OptimizerType::ScalesType ScalesType;
  typedef typename Superclass::DerivativeType            DerivativeType;
  
protected:

  TranslationThenRotationImageRegistrationMethod();
  virtual ~TranslationThenRotationImageRegistrationMethod() {};

  /** This is the method, that all multi-stage optimizers must implement. */
  virtual void DoMultiStageRegistration(OptimizerPointer optimizer, TransformPointer transform) throw (ExceptionObject);

private:
  
  TranslationThenRotationImageRegistrationMethod(const Self&); // purposefully not implemented
  void operator=(const Self&);                                 // purposefully not implemented

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTranslationThenRotationImageRegistrationMethod.txx"
#endif

#endif



