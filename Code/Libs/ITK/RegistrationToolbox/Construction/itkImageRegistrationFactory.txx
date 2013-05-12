/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageRegistrationFactory_txx
#define __itkImageRegistrationFactory_txx

#include <ConversionUtils.h>
#include "itkImageRegistrationFactory.h"
#include <itkTransformFactory.h>
#include <itkTransformFileReader.h>
#include <itkNIFTKTransformIOFactory.h>

#include <itkLogHelper.h>

namespace itk
{
template <typename TInputImageType, unsigned int Dimension, class TScalarType>
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::ImageRegistrationFactory()
{
  niftkitkDebugMacro(<<"ImageRegistrationFactory(): Constructed");

  itk::ObjectFactoryBase::RegisterFactory(itk::NIFTKTransformIOFactory::New());
}

template<typename TInputImageType, unsigned int Dimension, class TScalarType>
void 
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template<typename TInputImageType, unsigned int Dimension, class TScalarType>
typename ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::SingleResRegistrationType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateSingleResRegistration(SingleResRegistrationMethodTypeEnum type)
{
  
  typename SingleResRegistrationType::Pointer pointer;
  if (type == SINGLE_RES_MASKED)
    {
      niftkitkDebugMacro(<<"CreateSingleResRegistration(): Creating Masked Reg Method");
      pointer = SingleResRegistrationType::New();
    }
  else if (type == SINGLE_RES_TRANS_ROTATE)
    {
      niftkitkDebugMacro(<<"CreateSingleResRegistration(): Creating Translation Then Rotation Method");
      pointer = TranslationThenRotationRegistrationType::New();
    }  
  else if (type == SINGLE_RES_TRANS_ROTATE_SCALE)
    {
      niftkitkDebugMacro(<<"CreateSingleResRegistration(): Creating Translate Rotate Scale Method");
      pointer = TranslateRotateScaleRegistrationType::New();
    }  
  else if (type == SINGLE_RES_RIGID_SCALE)
    {
      niftkitkDebugMacro(<<"CreateSingleResRegistration(): Creating Rigid Plus Scale Method");
      pointer = RigidPlusScaleRegistrationType::New();
    }
  else if (type == SINGLE_RES_BLOCK_MATCH)
    {
      niftkitkDebugMacro(<<"CreateSingleResRegistration(): Creating a Block Matching method");
      pointer = BlockMatchingRegistrationType::New();  
    }
  else
    {
      itkExceptionMacro(<<"Unrecognised image registration method" << type);  
    }
  niftkitkDebugMacro(<<"CreateSingleResRegistration(): Returning object: " << &pointer);
  return pointer;
}

template<typename TInputImageType, unsigned int Dimension, class TScalarType>
typename ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::MultiResRegistrationType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateMultiResRegistration(MultiResRegistrationMethodTypeEnum type)
{
  typename MultiResRegistrationType::Pointer pointer;
  if (type == MULTI_RES_NORMAL)
    {
      niftkitkDebugMacro(<<"CreateMultiResRegistration(): Creating Multi Res Method");
      pointer = MultiResRegistrationType::New();
    }
  else
    {
      itkExceptionMacro(<<"Unrecognised multi-resolution image registration method" << type);  
    }
  niftkitkDebugMacro(<<"CreateMultiResRegistration(): Returning object: " << &pointer);
  return pointer;
}

template<typename TInputImageType, unsigned int Dimension, class TScalarType>
typename ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::InterpolatorType::Pointer 
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateInterpolator(InterpolationTypeEnum type)
{
  typename InterpolatorType::Pointer interp;
  
  if (type == NEAREST)
    {
      niftkitkDebugMacro(<<"CreateInterpolator(): Creating Nearest Neighbour interpolator");
      interp = NearestNeighbourInterpolatorType::New();
    }
  else if (type == LINEAR)
    {
      niftkitkDebugMacro(<<"CreateInterpolator(): Creating Linear interpolator");
      interp = LinearInterpolatorType::New();
    }
  else if (type == BSPLINE)
    {
      niftkitkDebugMacro(<<"CreateInterpolator(): Creating B-Spline interpolator");
      interp = BSplineInterpolatorType::New();
    }
  else if (type == SINC)
    {
      niftkitkDebugMacro(<<"CreateInterpolator(): Creating Sinc interpolator");
      interp = SincInterpolatorType::New();
    }
  else
    {
      itkExceptionMacro(<< "Unrecognised interpolation type: " << type);  
    }
  niftkitkDebugMacro(<<"CreateInterpolator(): Returning object: " << &interp);
  return interp;
}

template<typename TInputImageType, unsigned int Dimension, class TScalarType>
typename ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::MetricType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateMetric(MetricTypeEnum type)
{
  typename MetricType::Pointer metric;
  
  if (type == SSD)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating SSD metric");
      metric = SSDMetricType::New();
    }
  else if (type == MSD)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating MSD metric");
      metric = MSDMetricType::New();
    }  
  else if (type == SAD)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating SAD metric");
      metric = SADMetricType::New();
    }
  else if (type == NCC)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating NCC metric");
      metric = NCCMetricType::New();
    }  
  else if (type == RIU)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating RIU metric");
      metric = RIUMetricType::New();
    }  
  else if (type == PIU)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating PIU metric");
      metric = PIUMetricType::New();
    }
  else if (type == JE)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating JE metric");
      metric = JEMetricType::New();
    }
  else if (type == MI)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating MI metric");
      metric = MIMetricType::New();
    }
  else if (type == NMI)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating NMI metric");
      metric = NMIMetricType::New();
    }
  else if (type == CR)
    {
      niftkitkDebugMacro(<<"CreateMetric(): Creating CR metric");
      metric = CRMetricType::New();
    }      
  else
    {
      itkExceptionMacro(<< "Unrecognised metric type: " << type);  
    }
  niftkitkDebugMacro(<<"CreateMetric(): Returning object: " << &metric);
  return metric;
}
  
template<typename TInputImageType, unsigned int Dimension, class TScalarType>
typename ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::TransformType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateTransform(TransformTypeEnum type)
{
  typename TransformType::Pointer transform;

  if (type == TRANSLATION)
    {
      niftkitkDebugMacro(<<"CreateTransform(): Creating UCL Euler Translation transform");
      typename EulerAffineTransformType::Pointer ucltransform = EulerAffineTransformType::New();
      ucltransform->SetJustTranslation();
      transform = ucltransform;
    }
  else if (type == RIGID)
    {
      niftkitkDebugMacro(<<"CreateTransform(): Creating UCL Euler Rigid transform");
      typename EulerAffineTransformType::Pointer ucltransform = EulerAffineTransformType::New();
      ucltransform->SetRigid();
      transform = ucltransform;
    }
  else if (type == RIGID_SCALE)
    {
      niftkitkDebugMacro(<<"CreateTransform(): Creating UCL Euler Rigid Plus Scale transform");
      typename EulerAffineTransformType::Pointer ucltransform = EulerAffineTransformType::New();
      ucltransform->SetRigidPlusScale();
      transform = ucltransform;      
    }
  else if (type == AFFINE)
    {
      niftkitkDebugMacro(<<"CreateTransform(): Creating UCL Euler Affine transform");
      typename EulerAffineTransformType::Pointer ucltransform = EulerAffineTransformType::New();
      ucltransform->SetFullAffine();
      transform = ucltransform;      
    }  
  else 
    {
      itkExceptionMacro(<< "Unrecognised transform type: " << type);  
    }
  niftkitkDebugMacro(<<"CreateTransform(): Returning object: " << &transform << ", with: " << transform->GetNumberOfParameters() << " parameters");
  return transform;
}


template<typename TInputImageType, unsigned int Dimension, class TScalarType>
typename ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::TransformType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateTransform(std::string transformFilename)
{
  typename TransformType::Pointer transform;
  
  try 
  {
    itk::TransformFactory< PerspectiveProjectionTransformType >::RegisterTransform();
    itk::TransformFactory< EulerAffineTransformType >::RegisterTransform();
    itk::TransformFactory< FluidDeformableTransformType >::RegisterTransform();
    itk::TransformFactory< BSplineDeformableTransformType >::RegisterTransform(); 
    itk::TransformFactory< ITKAffineTransformType >::RegisterTransform();
    itk::TransformFactory< PCADeformationModelTransformType >::RegisterTransform();
    itk::TransformFactory< TranslationPCADeformationModelTransformType >::RegisterTransform();
  
    typedef itk::TransformFileReader TransformFileReaderType;
    TransformFileReaderType::Pointer transformFileReader = TransformFileReaderType::New();
    transformFileReader->SetFileName(transformFilename);
    transformFileReader->Update();
  
    typedef TransformFileReader::TransformListType* TransformListType;
    TransformListType transforms = transformFileReader->GetTransformList();
    niftkitkDebugMacro(<<"Number of transforms = " << transforms->size());
  
    typename itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
    for (; it != transforms->end(); ++it)
    {
      if (strcmp((*it)->GetNameOfClass(),"PerspectiveProjectionTransform") == 0)
      {
        transform = static_cast<TransformType*>((*it).GetPointer());
        niftkitkDebugMacro(<<"PerspectiveProjectionTransform found");
        transform->Print(std::cout);
        break;
      }
      else if (strcmp((*it)->GetNameOfClass(),"EulerAffineTransform") == 0)
      {
        transform = static_cast<TransformType*>((*it).GetPointer());
        niftkitkDebugMacro(<<"EulerAffineTransform found");
        transform->Print(std::cout);
        break;
      }
      else if (strcmp((*it)->GetNameOfClass(),"AffineTransform") == 0)
      {
        // Affine registration programs such as niftkBlockMatching and niftkAffine
        // will output standard AffineTransform classes. However, when we read
        // them back in, we should create our standard EulerAffineTransform, and
        // just use the Affine transform to initialize it.
        typename ITKAffineTransformType::Pointer tmpAffineTransform = static_cast<ITKAffineTransformType*>((*it).GetPointer());
        typename EulerAffineTransformType::Pointer tmpEulerTransform = EulerAffineTransformType::New();
        tmpEulerTransform->SetParametersFromTransform(tmpAffineTransform);
        transform = tmpEulerTransform;
        transform->Print(std::cout);
        break;
      }
      else if (strcmp((*it)->GetNameOfClass(),"BSplineTransform") == 0)
      {
        transform = static_cast<TransformType*>((*it).GetPointer());
        niftkitkDebugMacro(<<"BSplineTransform found");
      }
      else if (strcmp((*it)->GetNameOfClass(),"FluidDeformableTransform") == 0)
      {
        transform = static_cast<TransformType*>((*it).GetPointer());
        niftkitkDebugMacro(<<"FluidDeformableTransform found");
      }
      else if (strcmp((*it)->GetNameOfClass(),"PCADeformationModelTransform") == 0)
      {
        transform = static_cast<TransformType*>((*it).GetPointer());
        niftkitkDebugMacro(<<"PCADeformationModelTransform found");
      }
      else if (strcmp((*it)->GetNameOfClass(),"TranslationPCADeformationModelTransform") == 0)
      {
        transform = static_cast<TransformType*>((*it).GetPointer());
        niftkitkDebugMacro(<<"TranslationPCADeformationModelTransform found");
      }
    }
  }
  catch( itk::ExceptionObject & err ) 
  {
    niftkitkErrorMacro("Caught exception " << err);
    niftkitkDebugMacro(<<"Caught exception, so file " << transformFilename << " is not an ITK transform");
  }      

  if (transform.IsNull())
  {
    // Last resort, it might be a plain transformation text file, containing a 4x4 matrix    
    typename EulerAffineTransformType::Pointer tmpEulerTransform = EulerAffineTransformType::New();
    if (tmpEulerTransform->LoadFullAffineMatrix(transformFilename))
      {
        transform = tmpEulerTransform;
      }    
  }
  if (transform.IsNull())
  {
    itkExceptionMacro("No transform found."); 
  }
  
  return transform; 
}
    
template<typename TInputImageType, unsigned int Dimension, class TScalarType>
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::OptimizerType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateOptimizer(OptimizerTypeEnum type)
{
  typename OptimizerType::Pointer optimizer;
  
  if (type == SIMPLEX)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating SimplexType");
      optimizer = SimplexType::New();
    }
  else if (type == GRADIENT_DESCENT)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating GradientDescentType");
      optimizer = GradientDescentType::New();
    }
  else if (type == REGSTEP_GRADIENT_DESCENT)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating RegularStepGradientDescentType");
      optimizer = RegularStepGradientDescentType::New();
    }
  else if (type == CONJUGATE_GRADIENT_DESCENT)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating ConjugateGradientType");
      optimizer = ConjugateGradientType::New();
    }
  else if (type == POWELL)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating PowellOptimizerType");
      optimizer = PowellOptimizerType::New();
    }
  else if (type == SIMPLE_REGSTEP)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating UCLRegularStepOptimizerType");
      optimizer = UCLRegularStepOptimizerType::New();
    }
  else if (type == UCLPOWELL)
    {
      niftkitkDebugMacro(<<"CreateOptimizer(): Creating UCLPowellOptimizerType");
      optimizer = UCLPowellOptimizerType::New();
    }
  else
    {
      itkExceptionMacro(<< "Unrecognised optimizer type: " << type);  
    }
  niftkitkDebugMacro(<<"CreateOptimizer(): Returning object: " << &optimizer);
  return optimizer;
}


template<typename TInputImageType, unsigned int Dimension, class TScalarType>
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>::IterationUpdateCommandType::Pointer
ImageRegistrationFactory<TInputImageType, Dimension, TScalarType>
::CreateIterationUpdateCommand(OptimizerTypeEnum type)
{
  IterationUpdateCommandType::Pointer pointer;
  if (type == SIMPLEX || type == CONJUGATE_GRADIENT_DESCENT)
    {
      niftkitkDebugMacro(<<"Creating VnlIterationUpdateCommandType");
      pointer = VnlIterationUpdateCommandType::New();
    }
  else
    {
      niftkitkDebugMacro(<<"Creating IterationUpdateCommandType");
      pointer = IterationUpdateCommandType::New();
    }
  niftkitkDebugMacro(<<"CreateIterationUpdateCommand(): Returning object: " << &pointer);
  return pointer;
}

} // end namespace

#endif // __itkImageRegistrationFactory_txx
