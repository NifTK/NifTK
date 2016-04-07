/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkSingleResolutionImageRegistrationMethod_txx
#define _itkSingleResolutionImageRegistrationMethod_txx

#include <itkUCLMacro.h>
#include "itkSingleResolutionImageRegistrationMethod.h"


namespace itk
{
/*
 * Constructor
 */
template < typename TFixedImage, typename TMovingImage >
SingleResolutionImageRegistrationMethod<TFixedImage,TMovingImage>
::SingleResolutionImageRegistrationMethod()
: ImageRegistrationMethod<TFixedImage,TMovingImage>()
{
  niftkitkDebugMacro("ImageRegistrationMethod():Constructed");
}

/*
 * PrintSelf
 */
template < typename TFixedImage, typename TMovingImage >
void
SingleResolutionImageRegistrationMethod<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "IterationUpdate:" << m_IterationUpdateCommand << std::endl;
}

/**
 * Initialize by setting the interconnects between components. 
 */
template < typename TFixedImage, typename TMovingImage >
void
SingleResolutionImageRegistrationMethod<TFixedImage,TMovingImage>
::Initialize() throw (ExceptionObject)
{
  // Do the superclass initialisation. 
  Superclass::Initialize(); 
  
  typedef SimilarityMeasure<TFixedImage, TMovingImage> MetricType; 
  MetricType* metric = dynamic_cast<MetricType*>(this->GetMetric()); 
  
  if (metric != NULL)
  {
    // Set the interpolators. 
    if (this->m_FixedImageInterpolator)
      metric->SetFixedImageInterpolator(m_FixedImageInterpolator);
    if (this->m_MovingImageInterpolator)
      metric->SetMovingImageInterpolator(m_MovingImageInterpolator);
  }
}


/*
 * Generate Data
 */
template < typename TFixedImage, typename TMovingImage >
void
SingleResolutionImageRegistrationMethod<TFixedImage,TMovingImage>
::GenerateData()
{

  ParametersType empty(1);
  empty.Fill( 0.0 );
  try
    {
      // initialize the interconnects between components      
      this->Initialize();
      this->SetFixedImageRegion(this->GetFixedImage()->GetBufferedRegion());      
      if (m_IterationUpdateCommand.GetPointer() != 0)
        {
          this->GetOptimizer()->AddObserver( itk::IterationEvent(), m_IterationUpdateCommand );
        }  
      niftkitkDebugMacro("Done SingleResolutionImageRegistrationMethod::Initialize()");
    }
  catch( ExceptionObject& err )
    {
       this->SetLastTransformParameters(empty);
       throw err;
    }

  try
    {
      // do the optimization
      this->DoRegistration();
    }
  catch( ExceptionObject& err )
    {
      this->SetLastTransformParameters(this->GetOptimizer()->GetCurrentPosition());

      // Pass exception to caller
      throw err;
    }
}

/*
 * The optimize bit that we can now override.
 */
template < typename TFixedImage, typename TMovingImage >
void
SingleResolutionImageRegistrationMethod<TFixedImage,TMovingImage>
::DoRegistration() throw (ExceptionObject)
{
  niftkitkDebugMacro("DoRegistration():Start");
  
  this->GetOptimizer()->StartOptimization(); 
  
  niftkitkDebugMacro("DoRegistration():Copying back to LastTransformParameters.");
  
  this->SetLastTransformParameters(this->GetOptimizer()->GetCurrentPosition());
  
  niftkitkDebugMacro("DoRegistration():Setting it onto the transformation, which is object:" << this->GetTransform());
  
  this->GetTransform()->SetParameters(this->GetOptimizer()->GetCurrentPosition());
  
  niftkitkDebugMacro("Optimizer position:" << this->GetOptimizer()->GetCurrentPosition());
  niftkitkDebugMacro("DoRegistration():Finished");
}

} // end namespace itk


#endif
