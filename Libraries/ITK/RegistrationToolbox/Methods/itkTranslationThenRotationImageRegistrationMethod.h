/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkTranslationThenRotationImageRegistrationMethod_h
#define itkTranslationThenRotationImageRegistrationMethod_h


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



