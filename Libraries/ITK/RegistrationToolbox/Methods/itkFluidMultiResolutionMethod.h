/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkFluidMultiResolutionMethod_h
#define itkFluidMultiResolutionMethod_h

#include "itkMultiResolutionDeformableImageRegistrationMethod.h"
#include <itkUCLBSplineTransform.h>

namespace itk
{

/** 
 * \class FluidMultiResolutionMethod
 * \brief Extends MultiResolutionDeformableImageRegistrationMethod to sort out interpolating
 * the deformation fluid in between resolution levels.
 * 
 * \sa MultiResolutionDeformableImageRegistrationMethod
 */
template <typename TInputImageType, class TScalarType, unsigned int NDimensions, class TPyramidFilter >
class ITK_EXPORT FluidMultiResolutionMethod : 
public MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, float, TPyramidFilter> 
{
public:
  /** Standard class typedefs. */
  typedef FluidMultiResolutionMethod                                         Self;
  typedef MultiResolutionDeformableImageRegistrationMethod<TInputImageType, TScalarType, NDimensions, float> Superclass;
  typedef SmartPointer<Self>                                                 Pointer;
  typedef SmartPointer<const Self>                                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FluidMultiResolutionMethod, MultiResolutionDeformableImageRegistrationMethod);

  /**  Type of the input image. */
  typedef TInputImageType                                                    InputImageType;
  typedef typename InputImageType::SpacingType                               InputImageSpacingType;

  /**  Type of the Transform . */
  typedef FluidDeformableTransform<TInputImageType, TScalarType, NDimensions, float >         FluidDeformableTransformType;
  typedef typename FluidDeformableTransformType::Pointer                             FluidDeformableTransformPointer;

  typedef FluidGradientDescentOptimizer<TInputImageType, TInputImageType, double, float> OptimizerType;
  typedef OptimizerType*                                                     OptimizerPointer;

  /** Set/Get the Transfrom. */
  itkSetObjectMacro( Transform, FluidDeformableTransformType );
  itkGetObjectMacro( Transform, FluidDeformableTransformType );
  
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
  FluidMultiResolutionMethod();
  virtual ~FluidMultiResolutionMethod() {};

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
  typename FluidDeformableTransformType::DeformableParameterPointerType m_InitialDeformableTransformParametersOfNextLevel; 

private:
  FluidMultiResolutionMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  FluidDeformableTransformPointer m_Transform;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFluidMultiResolutionMethod.txx"
#endif

#endif



