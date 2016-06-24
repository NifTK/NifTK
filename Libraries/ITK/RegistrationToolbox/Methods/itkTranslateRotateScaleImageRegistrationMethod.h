/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkTranslateRotateScaleImageRegistrationMethod_h
#define itkTranslateRotateScaleImageRegistrationMethod_h


#include "itkMultiStageImageRegistrationMethod.h"

namespace itk
{

/** 
 * \class TranslateRotateScaleImageRegistrationMethod
 * \brief Class specifically for doing Translate, Rotate, Scale registration.
 *
 * The aim is to interleave:
 * 
 * 1. Transations only, using brain image.
 * 
 * 2. Rotations only, using brain image.
 * 
 * 3. Scale only, using dilated brain (i.e. brain plus skull, scalp).
 * 
 *  
 * \ingroup RegistrationFilters
 */
template <typename TInputImageType>
class ITK_EXPORT TranslateRotateScaleImageRegistrationMethod 
: public MultiStageImageRegistrationMethod<TInputImageType> 
{
public:

  /** Standard class typedefs. */
  typedef TranslateRotateScaleImageRegistrationMethod                  Self;
  typedef MultiStageImageRegistrationMethod<TInputImageType>           Superclass;
  typedef SmartPointer<Self>                                           Pointer;
  typedef SmartPointer<const Self>                                     ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(TranslateRotateScaleImageRegistrationMethod, MultiStageImageRegistrationMethod);

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

  TranslateRotateScaleImageRegistrationMethod();
  virtual ~TranslateRotateScaleImageRegistrationMethod() {};

  /** This is the method, that all multi-stage optimizers must implement. */
  virtual void DoMultiStageRegistration(OptimizerPointer optimizer, TransformPointer transform) throw (ExceptionObject);

private:
  
  TranslateRotateScaleImageRegistrationMethod(const Self&); // purposefully not implemented
  void operator=(const Self&);                              // purposefully not implemented

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTranslateRotateScaleImageRegistrationMethod.txx"
#endif

#endif



