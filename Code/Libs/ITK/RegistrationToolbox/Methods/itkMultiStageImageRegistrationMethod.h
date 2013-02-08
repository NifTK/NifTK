/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMultiStageImageRegistrationMethod_h
#define __itkMultiStageImageRegistrationMethod_h


#include "itkMaskedImageRegistrationMethod.h"
#include "itkImageToImageMetric.h"
#include "itkEulerAffineTransform.h"
#include "itkUCLRegularStepOptimizer.h"

namespace itk
{

/** 
 * \class MultiStageImageRegistrationMethod
 * \brief Base Class specifically for doing multiple stage registrations.
 * 
 * The aim is so that we can test registration methods that do things like:
 * 
 * 1.) Optimize translations then rotations.
 * 
 * 2.) Optimize translations then rotations then scales.
 * 
 * 3.) Optimize rigid parameters (translations & rotations) then scales.
 * 
 * We have an initial step size.
 * We loop round a combination such as translate, rotate, scale, calling StartOptimization for each.
 * We keep looping until neither translation, rotation and scale have less change on cost function than the tolerance.
 * We then reduce the step size and try again.
 * \ingroup RegistrationFilters
 */
template <typename TInputImageType>
class ITK_EXPORT MultiStageImageRegistrationMethod 
: public MaskedImageRegistrationMethod<TInputImageType> 
{
public:

  /** Standard class typedefs. */
  typedef MultiStageImageRegistrationMethod                        Self;
  typedef MaskedImageRegistrationMethod<TInputImageType>           Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiStageImageRegistrationMethod, MaskedImageRegistrationMethod);

  /** Typedefs. */
  typedef ImageToImageMetric< TInputImageType,
                              TInputImageType >                         MetricType;
  typedef typename MetricType::TransformParametersType                  ParametersType;
  typedef typename MetricType::DerivativeType                           DerivativeType;                                   
  
  /** Subclasses all RELY on this transformation. */
  typedef itk::EulerAffineTransform<double, 
                                       TInputImageType::ImageDimension,
                                       TInputImageType::ImageDimension> TransformType;
  typedef TransformType*                                                TransformPointer;

  /** Subclasses all RELY on this optimizer. */
  typedef UCLRegularStepOptimizer                                       OptimizerType;
  typedef OptimizerType*                                                OptimizerPointer;
  typedef OptimizerType::ScalesType                                     ScalesType;


protected:

  MultiStageImageRegistrationMethod();
  virtual ~MultiStageImageRegistrationMethod() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** This is called by base class. */
  virtual void DoRegistration() throw (ExceptionObject);

  /** This is the method, that all multi-stage optimizers must implement. */
  virtual void DoMultiStageRegistration(OptimizerPointer optimizer, TransformPointer transform) throw (ExceptionObject) {};
  
  /** Set the tolerance. */
  itkSetMacro( LoopTolerance, double );
  itkGetMacro( LoopTolerance, double );

  /** Maximum times round the loop */
  itkSetMacro( MaxNumberOfLoops, unsigned int );
  itkGetMacro( MaxNumberOfLoops, unsigned int );

  /** Set the reduction factor. */
  itkSetMacro( LoopStepSizeReductionFactor, double);
  itkGetMacro( LoopStepSizeReductionFactor, double);
  
private:
  
  MultiStageImageRegistrationMethod(const Self&); // purposefully not implemented
  void operator=(const Self&);                    // purposefully not implemented

  /** So we can stop the registration before phases. Default 5.*/
  unsigned int m_MaxNumberOfLoops;
  
  /** So we can stop the registration between phases. Default 0.01 */
  double m_LoopTolerance;
  
  /** So we can reduce the step size at the end of each loop. */
  double m_LoopStepSizeReductionFactor;
};


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiStageImageRegistrationMethod.txx"
#endif

#endif



