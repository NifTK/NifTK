/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVelocityFieldMultiResolutionMethod_h
#define __itkVelocityFieldMultiResolutionMethod_h

#include "itkMultiResolutionDeformableImageRegistrationMethod.h"
#include <itkBSplineTransform.h>
#include <itkVelocityFieldDeformableTransform.h>

namespace itk
{

/** 
 * \class VelocityFieldMultiResolutionMethod
 * \brief Extends MultiResolutionDeformableImageRegistrationMethod to sort out interpolating
 * the deformation fluid in between resolution levels.
 * 
 * \sa MultiResolutionDeformableImageRegistrationMethod
 */
template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TPyramidFilter >
class VelocityFieldMultiResolutionMethod : 
public MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, float, TPyramidFilter> 
{
public:
  /** Standard class typedefs. */
  typedef VelocityFieldMultiResolutionMethod                                         Self;
  typedef MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, float> Superclass;
  typedef SmartPointer<Self>                                                 Pointer;
  typedef SmartPointer<const Self>                                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VelocityFieldMultiResolutionMethod, MultiResolutionDeformableImageRegistrationMethod);

  /**  Type of the input image. */
  typedef TInputImageType                                                    InputImageType;
  typedef typename InputImageType::SpacingType                               InputImageSpacingType;

  /**  Type of the Transform . */
  typedef VelocityFieldDeformableTransform<TInputImageType, TScalarType, NDimensions, float >         VelocityFieldDeformableTransformType;
  typedef typename VelocityFieldDeformableTransformType::Pointer                             VelocityFieldDeformableTransformPointer;

  typedef VelocityFieldGradientDescentOptimizer<TInputImageType, TInputImageType, double, float> OptimizerType;
  typedef OptimizerType*                                                     OptimizerPointer;

  /** Set/Get the Transfrom. */
  itkSetObjectMacro( Transform, VelocityFieldDeformableTransformType );
  itkGetObjectMacro( Transform, VelocityFieldDeformableTransformType );
  
  /**
   * The max step size in voxel unit. 
   * 
   * and indicates the maximum step size taken per iteration.
   * Default is 5.0, so the initial step size will equal the maximum voxel dimension.
   * Then as the registration progresses, this is reduced accordingly.
   */
  itkSetMacro(MaxStepSize, double);
  itkGetMacro(MaxStepSize, double);

  /**
   * 
   */
  itkSetMacro(MinDeformationSize, double);
  itkGetMacro(MinDeformationSize, double);
  
  /** Method that initiates the registration. */
  virtual void StartRegistration();
  
protected:
  VelocityFieldMultiResolutionMethod();
  virtual ~VelocityFieldMultiResolutionMethod() {};

  /**
   * In here, we sort out initialising the BSpline grid at first resolution
   * level, and then interpolating the BSpline grid inbetween resolution levels.
   */
  virtual void BeforeSingleResolutionRegistration();
  
  /** to calculate a max step size. */
  double m_MaxStepSize;
  
  /** to calculate a max step size. */
  double m_MinDeformationSize;
  
  /**
   * Initial deformable parameters. 
   */
  typename VelocityFieldDeformableTransformType::DeformableParameterPointerType m_InitialDeformableTransformParametersOfNextLevel; 

private:
  VelocityFieldMultiResolutionMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  VelocityFieldDeformableTransformPointer m_Transform;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVelocityFieldMultiResolutionMethod.txx"
#endif

#endif



