/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRigidPlusScaleImageRegistrationMethod_h
#define __itkRigidPlusScaleImageRegistrationMethod_h


#include "itkMultiStageImageRegistrationMethod.h"

namespace itk
{

/** 
 * \class RigidPlusScaleImageRegistrationMethod
 * \brief Class specifically for doing Rigid plus Scale registration.
 *
 * The aim is to interleave:
 * 
 * 1. Rigid, using brain image
 * 
 * 2. Scale only, using dilated brain (i.e. brain plus skull, scalp).
 *
 *  
 * \ingroup RegistrationFilters
 */
template <typename TInputImageType>
class ITK_EXPORT RigidPlusScaleImageRegistrationMethod 
: public MultiStageImageRegistrationMethod<TInputImageType> 
{
public:

  /** Standard class typedefs. */
  typedef RigidPlusScaleImageRegistrationMethod                        Self;
  typedef MultiStageImageRegistrationMethod<TInputImageType>           Superclass;
  typedef SmartPointer<Self>                                           Pointer;
  typedef SmartPointer<const Self>                                     ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(RigidPlusScaleImageRegistrationMethod, MultiStageImageRegistrationMethod);

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

  RigidPlusScaleImageRegistrationMethod();
  virtual ~RigidPlusScaleImageRegistrationMethod() {};

  /** This is the method, that all multi-stage optimizers must implement. */
  virtual void DoMultiStageRegistration(OptimizerPointer optimizer, TransformPointer transform) throw (ExceptionObject);

private:
  
  RigidPlusScaleImageRegistrationMethod(const Self&); // purposely not implemented
  void operator=(const Self&);                                        // purposely not implemented

};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRigidPlusScaleImageRegistrationMethod.txx"
#endif

#endif



