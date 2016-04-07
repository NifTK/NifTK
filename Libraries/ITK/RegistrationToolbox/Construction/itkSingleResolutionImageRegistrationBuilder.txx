/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkSingleResolutionImageRegistrationBuilder_txx
#define _itkSingleResolutionImageRegistrationBuilder_txx

#include <itkLogHelper.h>
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include <itkContinuousIndex.h>

namespace itk
{
/*
 * Constructor
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::SingleResolutionImageRegistrationBuilder()
{
  m_ImageRegistrationFactory = ImageRegistrationFactoryType::New();
  niftkitkDebugMacro(<<"SingleResolutionImageRegistrationBuilder():Constructed");
}


/*
 * PrintSelf
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
void
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << m_ImageRegistrationFactory << std::endl;
}

/**
 * Start the Creation Process.
 */
template < typename TImage, unsigned int Dimension, class TScalarType>
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::SingleResRegType::Pointer
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::StartCreation(SingleResRegistrationMethodTypeEnum type)
{
  
  typename SingleResRegType::Pointer regmethod = m_ImageRegistrationFactory->CreateSingleResRegistration(type);
  
  m_ImageRegistrationMethod = regmethod;
  m_ImageRegistrationMethodEnum = type;

  niftkitkDebugMacro(<<"StartCreation():Created an itkSingleResolutionImageRegistrationMethod:" << m_ImageRegistrationMethod.GetPointer() << ", type:" << m_ImageRegistrationMethodEnum);
  
  return regmethod;
}

/**
 * Create the interpolator.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::InterpolatorType::Pointer 
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateInterpolator(InterpolationTypeEnum type)
{
  typename InterpolatorType::Pointer interpolator = m_ImageRegistrationFactory->CreateInterpolator(type);
  
  m_ImageRegistrationMethod->SetInterpolator(interpolator);
  m_InterpolatorEnum = type;

  niftkitkDebugMacro(<<"CreateInterpolator():Created an Interpolator:" << interpolator.GetPointer() << ", of type:" << m_InterpolatorEnum);
    
  return interpolator;
}

/**
 * Create the metric.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::MetricType::Pointer
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateMetric(MetricTypeEnum type)
{
  typename MetricType::Pointer metric = m_ImageRegistrationFactory->CreateMetric(type);
  
  m_ImageRegistrationMethod->SetMetric(metric);
  m_MetricEnum = type;

  niftkitkDebugMacro(<<"CreateMetric():Created a Metric:" << metric.GetPointer() << ", of type:" << m_MetricEnum);
  
  return metric;
}

/**
 * Create the transform.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::TransformType::Pointer
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateTransform(TransformTypeEnum type, ImageConstPointer fixedImage)
{
  niftkitkDebugMacro(<<"CreateTransform():Creating a Transform of type:" << type);
  
  typename TransformType::Pointer transform;
  
  if (m_ImageRegistrationMethodEnum == SINGLE_RES_RIGID_SCALE
    || m_ImageRegistrationMethodEnum == SINGLE_RES_TRANS_ROTATE
    || m_ImageRegistrationMethodEnum == SINGLE_RES_TRANS_ROTATE_SCALE
    || type == RIGID || type == RIGID_SCALE || type == AFFINE)
    {
      if ( m_ImageRegistrationMethodEnum == SINGLE_RES_RIGID_SCALE 
        || m_ImageRegistrationMethodEnum == SINGLE_RES_TRANS_ROTATE
        || m_ImageRegistrationMethodEnum == SINGLE_RES_TRANS_ROTATE_SCALE)
        {
          niftkitkWarningMacro("CreateTransform():Forcing transformation to: AFFINE");
      
          transform = m_ImageRegistrationFactory->CreateTransform(AFFINE);
        }
      else
        {
          transform = m_ImageRegistrationFactory->CreateTransform(type);  
        }  

      EulerAffineTransformPointer trans = dynamic_cast<EulerAffineTransformPointer>(transform.GetPointer());
      
      trans->SetIdentity(); 
      // want to rotate about centre of image.
      ImageRegionType region = fixedImage->GetLargestPossibleRegion();
      ImageSizeType size = region.GetSize();
      
      ContinuousIndex<double, TImage::ImageDimension> center;
  
      for (unsigned int i = 0; i < Dimension; i++)
        {
          center.SetElement(i, (size[i] - 1) / 2);
        }
      
      typename TImage::PointType centerPoint; 
      fixedImage->TransformContinuousIndexToPhysicalPoint(center, centerPoint);
      trans->SetCenter(centerPoint);
      
      niftkitkDebugMacro(<<"CreateTransform():Given fixed image size:" << size << ", spacing:" << fixedImage->GetSpacing() << ", set center to:" << center);
    }
  else
    {
      transform = m_ImageRegistrationFactory->CreateTransform(type);
    }
  
  m_ImageRegistrationMethod->SetTransform(transform);
  m_TransformEnum = type;
  
  niftkitkDebugMacro(<<"CreateTransform():Created a transform:" << transform.GetPointer() << ", of type:" << m_TransformEnum);
  
  return transform;
}

template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::TransformType::Pointer
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateTransform(std::string initialTransformName)
{
  typename TransformType::Pointer transform = m_ImageRegistrationFactory->CreateTransform(initialTransformName); 
  
  m_ImageRegistrationMethod->SetTransform(transform);
  niftkitkDebugMacro(<<"CreateTransform():Created a transform:" << transform.GetPointer() << ", from file:" << initialTransformName);
  
  return transform; 
}


/**
 * Create the optimizer.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::OptimizerType::Pointer
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateOptimizer(
  OptimizerTypeEnum type)
{
  niftkitkDebugMacro(<<"CreateOptimizer():Creating an Optimizer of type:" << type);
  
  typename OptimizerType::Pointer optimizer; 
  
  m_OptimizerEnum = type;
  
  if (m_ImageRegistrationMethodEnum == SINGLE_RES_RIGID_SCALE
    || m_ImageRegistrationMethodEnum == SINGLE_RES_TRANS_ROTATE
    || m_ImageRegistrationMethodEnum == SINGLE_RES_TRANS_ROTATE_SCALE
    || type == SIMPLE_REGSTEP)
    {
      if (type == SIMPLE_REGSTEP)
        {
          niftkitkDebugMacro(<<"CreateOptimizer():Setting optimizer to: SIMPLE_REGSTEP");
        }
      else
        {
          niftkitkWarningMacro("CreateOptimizer():Forcing optimizer to: SIMPLE_REGSTEP");
        }
      
      
      optimizer = m_ImageRegistrationFactory->CreateOptimizer(SIMPLE_REGSTEP);
      
      m_OptimizerEnum = SIMPLE_REGSTEP;
    }
  else
    {
      optimizer = m_ImageRegistrationFactory->CreateOptimizer(type);
    }
  
  m_ImageRegistrationMethod->SetOptimizer(optimizer);
  m_ImageRegistrationMethod->SetIterationUpdateCommand(m_ImageRegistrationFactory->CreateIterationUpdateCommand(m_OptimizerEnum));
      
  niftkitkDebugMacro(<<"CreateOptimizer():Created an optimizer:" << m_ImageRegistrationMethod->GetOptimizer() << ", of type:" << m_OptimizerEnum);
     
  return optimizer;
}
    
/**
 * Create the optimizer.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::SingleResRegType::Pointer
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::GetSingleResolutionImageRegistrationMethod()
{
  niftkitkDebugMacro(<<"GetSingleResolutionImageRegistrationMethod():Returning built object:" << m_ImageRegistrationMethod.GetPointer());
  return m_ImageRegistrationMethod;
}

/**
 * Create the interpolator.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::InterpolatorType::Pointer 
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateFixedImageInterpolator(InterpolationTypeEnum type)
{
  typename InterpolatorType::Pointer interpolator = m_ImageRegistrationFactory->CreateInterpolator(type);
  
  m_ImageRegistrationMethod->SetFixedImageInterpolator(interpolator);
  m_InterpolatorEnum = type;

  niftkitkDebugMacro(<<"CreateFixedImageInterpolator():Created an Interpolator:" << interpolator.GetPointer() << ", of type:" << m_InterpolatorEnum);
    
  return interpolator;
}

/**
 * Create the interpolator.
 */
template < typename TImage, unsigned int Dimension, class TScalarType >
typename SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>::InterpolatorType::Pointer 
SingleResolutionImageRegistrationBuilder<TImage, Dimension, TScalarType>
::CreateMovingImageInterpolator(InterpolationTypeEnum type)
{
  typename InterpolatorType::Pointer interpolator = m_ImageRegistrationFactory->CreateInterpolator(type);
  
  m_ImageRegistrationMethod->SetMovingImageInterpolator(interpolator);
  m_InterpolatorEnum = type;

  niftkitkDebugMacro(<<"CreateMovingImageInterpolator():Created an Interpolator:" << interpolator.GetPointer() << ", of type:" << m_InterpolatorEnum);
    
  return interpolator;
}




} // end namespace itk


#endif
