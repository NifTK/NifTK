/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKFFDGradientDescentOptimizer_TXX_
#define ITKFFDGradientDescentOptimizer_TXX_

#include "itkFFDGradientDescentOptimizer.h"
#include "itkLogHelper.h"
#include "itkUCLMacro.h"

namespace itk
{
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
FFDGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::FFDGradientDescentOptimizer()
{
  
  m_WriteForceImage = false;
  m_ForceImageFileName = "tmp.force";
  m_ForceImageFileExt = "vtk";
  
  m_ScaleVectorFieldFilter = ScaleFieldType::New();  
  m_GradientImageFilter = GradientFilterType::New();
  
  m_MinimumGradientVectorMagnitudeThreshold = 0.0000001;
  m_ScaleForceVectorsByGradientImage = false;
  m_ScaleByComponents = false;
  m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel = true;
  m_CalculateNextStepCounter = 0;
  
  niftkitkDebugMacro(<< "FFDGradientDescentOptimizer():Constructed, m_MinimumGradientVectorMagnitudeThreshold=" << m_MinimumGradientVectorMagnitudeThreshold \
    << ", m_ScaleForceVectorsByGradientImage:" << m_ScaleForceVectorsByGradientImage \
    << ", m_ScaleByComponents:" << m_ScaleByComponents \
    << ", m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel:" << m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel \
    << ", m_CalculateNextStepCounter=" << m_CalculateNextStepCounter \
      );
}

/*
 * PrintSelf
 */
template < typename TFixedImage, typename TMovingImage, class TScalarType, class TDeformationScalar >
void
FFDGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  
  if (!m_ForceFilter.IsNull())
    {
      os << indent << "ForceFilter=" << m_ForceFilter << std::endl;
    }
    
  if (!m_SmoothFilter.IsNull())
    {
      os << indent << "SmoothFilter=" << m_SmoothFilter << std::endl;
    }
    
  if (!m_InterpolatorFilter.IsNull())
    {
      os << indent << "InterpolatorFilter=" << m_InterpolatorFilter << std::endl;
    }

  if (!m_ScaleVectorFieldFilter.IsNull())
    {
      os << indent << "ScaleVectorFieldFilter=" << m_ScaleVectorFieldFilter << std::endl;
    }

  if (!m_GradientImageFilter.IsNull())
    {
      os << indent << "GradientImageFilter=" << m_GradientImageFilter << std::endl;
    }
    
  os << indent << "MinimumGradientVectorMagnitudeThreshold=" << m_MinimumGradientVectorMagnitudeThreshold << std::endl;
  os << indent << "ScaleForceVectorsByGradientImage=" << m_ScaleForceVectorsByGradientImage << std::endl;
  os << indent << "ScaleByComponents=" << m_ScaleByComponents << std::endl;
  os << indent << "SmoothGradientVectorsBeforeInterpolatingToControlPointLevel=" << m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel << std::endl;
  os << indent << "CalculateNextStepCounter=" << m_CalculateNextStepCounter << std::endl;
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::GetGradient(int iterationNumber, const ParametersType& current, ParametersType& next)
{
  niftkitkDebugMacro(<< "GetGradient():Started");
  
  if (m_ForceFilter.IsNull())
    {
      niftkitkExceptionMacro(<< "Force filter is not set");
    } 
  
  if (m_SmoothFilter.IsNull())
    {
      niftkitkExceptionMacro(<< "Smooth filter is not set");
    } 
  
  if (m_InterpolatorFilter.IsNull())
    {
      niftkitkExceptionMacro(<< "Interpolator filter is not set");
    } 

  BSplineTransformPointer transform = dynamic_cast<BSplineTransformPointer>(this->m_DeformableTransform.GetPointer());
  if (transform == 0)
    {
      niftkitkExceptionMacro(<< "Can't dynamic cast to BSplineTransform");
    }

  GridImagePointer gridPointer = transform->GetGrid();
      
  // Check the grid is the right size for the parameters.
  
  OutputImageSizeType size = gridPointer->GetLargestPossibleRegion().GetSize();
  unsigned long int numberOfGridVoxels = 1;
  unsigned long int expectedNumberOfParameters = Dimension;
  for (int i = 0; i < Dimension; i++)
    {
      expectedNumberOfParameters *= size[i];
      numberOfGridVoxels *= size[i];
    }  
  niftkitkDebugMacro(<< "GetGradient():Number of voxels=" << numberOfGridVoxels << ", expected parameters:" <<  expectedNumberOfParameters);
  
  if ( expectedNumberOfParameters != current.GetSize())
    {
      niftkitkExceptionMacro(<< "Grid size:" << expectedNumberOfParameters << " doesn't match current parameters array:" << current.GetSize());
    }

  if ( expectedNumberOfParameters != next.GetSize())
    {
      niftkitkExceptionMacro(<< "Grid size:" << expectedNumberOfParameters << " doesn't match next parameters array:" << next.GetSize());
    }

  niftkitkDebugMacro(<< "GetGradient():Grid spacing=" << gridPointer->GetSpacing());
  OutputImageSpacingType spacing;
  for (unsigned int i = 0; i < Dimension; i++)
    {
      spacing[i] = ((double)(this->m_FixedImage->GetLargestPossibleRegion().GetSize()[i])) / ((double)(size[i] -1));
    }
  niftkitkDebugMacro(<< "GetGradient():Voxel spacing=" << spacing);

  // Set the current parameter/deformation. 
  transform->SetParameters(current);

  niftkitkDebugMacro(<< "GetGradient():Fixed image at address=" << this->m_FixedImage \
      << ", size=" << this->m_FixedImage->GetLargestPossibleRegion().GetSize() \
      << ", spacing=" << this->m_FixedImage->GetSpacing() \
      << ", origin=" << this->m_FixedImage->GetOrigin() \
      << ", direction=" << this->m_FixedImage->GetDirection() \
      << ", transformed moving image at address=" << this->m_ImageToImageMetric->GetTransformedMovingImage() \
      << ", size:" << this->m_ImageToImageMetric->GetTransformedMovingImage()->GetLargestPossibleRegion().GetSize() \
      << ", spacing=" << this->m_ImageToImageMetric->GetTransformedMovingImage()->GetSpacing() \
      << ", origin=" << this->m_ImageToImageMetric->GetTransformedMovingImage()->GetOrigin() \
      << ",  direction=" << this->m_ImageToImageMetric->GetTransformedMovingImage()->GetDirection() \
      );
  
  m_ForceFilter->SetFixedImage(this->m_FixedImage);
  m_ForceFilter->SetTransformedMovingImage(this->m_ImageToImageMetric->GetTransformedMovingImage());
  m_ForceFilter->SetUnTransformedMovingImage(this->m_MovingImage);
  m_ForceFilter->SetFixedImageMask(this->m_ImageToImageMetric->GetFixedImageMask()); 
  m_ForceFilter->Modified();
  m_ForceFilter->UpdateLargestPossibleRegion();
  
  m_GradientImageFilter->SetInput(this->m_ImageToImageMetric->GetTransformedMovingImage());
  m_GradientImageFilter->Modified();
  m_GradientImageFilter->UpdateLargestPossibleRegion();
  
  m_ScaleVectorFieldFilter->SetImageThatWillBeScaled(m_ForceFilter->GetOutput());
  m_ScaleVectorFieldFilter->SetImageThatDeterminesTheAmountOfScaling(m_GradientImageFilter->GetOutput());
  m_ScaleVectorFieldFilter->SetScaleByComponents(m_ScaleByComponents);
  m_ScaleVectorFieldFilter->Modified();
  m_ScaleVectorFieldFilter->UpdateLargestPossibleRegion();
  
  // Wire smoothing filter to right components
  if (m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel)
    {
      if (m_ScaleForceVectorsByGradientImage)
        {
          niftkitkDebugMacro(<< "GetGradient():Connecting m_SmoothFilter to m_ScaleVectorFieldFilter");
          m_SmoothFilter->SetInput(m_ScaleVectorFieldFilter->GetOutput());  
        }
      else
        {
          niftkitkDebugMacro(<< "GetGradient():Connecting m_SmoothFilter to m_ForceFilter");
          m_SmoothFilter->SetInput(m_ForceFilter->GetOutput());
        }
      m_SmoothFilter->SetGridSpacing(spacing);
      m_SmoothFilter->Modified();  
      m_SmoothFilter->UpdateLargestPossibleRegion();
    }
    
  // Wire interpolating filter to right component.
  if (m_SmoothGradientVectorsBeforeInterpolatingToControlPointLevel)
    {
      niftkitkDebugMacro(<< "GetGradient():Connecting m_InterpolatorFilter to m_SmoothFilter");
      m_InterpolatorFilter->SetInterpolatedField(m_SmoothFilter->GetOutput());
    }
  else
    {
      if (m_ScaleForceVectorsByGradientImage)
        {
          niftkitkDebugMacro(<< "GetGradient():Connecting m_InterpolatorFilter to m_ScaleVectorFieldFilter");
          m_InterpolatorFilter->SetInterpolatedField(m_ScaleVectorFieldFilter->GetOutput());
        }
      else
        {
          niftkitkDebugMacro(<< "GetGradient():Connecting m_InterpolatorFilter to m_ForceFilter");
          m_InterpolatorFilter->SetInterpolatedField(m_ForceFilter->GetOutput());
        }
    }
  m_InterpolatorFilter->SetInterpolatingField(gridPointer);
  m_InterpolatorFilter->Modified();
  m_InterpolatorFilter->UpdateLargestPossibleRegion();

  OutputImagePointer output = m_InterpolatorFilter->GetOutput();    

  if (m_WriteForceImage)
    {
      std::string tmpFilename = m_ForceImageFileName + "." + niftk::ConvertToString((int)m_CalculateNextStepCounter) + "." + m_ForceImageFileExt;
      m_ForceFilter->WriteForceImage(tmpFilename);      
    }
    
  // Additionally, we may need constraint gradient.
  DerivativeType constraintDerivative(expectedNumberOfParameters);
  constraintDerivative.Fill(0); 
  
  if (this->m_ImageToImageMetric->GetUseConstraintGradient())
    {
      niftkitkDebugMacro(<< "GetGradient():Fetching constraint gradient, (very expensive)");
      this->m_ImageToImageMetric->GetConstraintDerivative(current, constraintDerivative);
    }
    
  OutputImagePixelType value;
  unsigned long int parameterIndex = 0;
  unsigned long int dimensionIndex = 0;
  
  niftkitkDebugMacro(<< "GetGradient():iterating over size:" << output->GetLargestPossibleRegion().GetSize() << " to copy vector image to array");
  OutputImageIteratorType iterator(output, output->GetLargestPossibleRegion());
  iterator.GoToBegin();
  while(!iterator.IsAtEnd())
    {
      value = iterator.Get();
      
//      printf("Matt: counter=%d, dx=%f, dy=%f, dz=%f\n",controlPointIndex, value[0], value[1], value[2]);
//      controlPointIndex++;
      
      for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
        {
          next.SetElement(parameterIndex, value[dimensionIndex] - constraintDerivative.GetElement(parameterIndex)); 
          parameterIndex++;
        }
      ++iterator;      
    }  
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
bool
FFDGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::LineAscent(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next)
{
  niftkitkDebugMacro(<< "LineAscent():Started, with current value=" << this->m_Value );

  // Iterate over each gradient vector to calculate maximum length of all vectors
  double length = 0;
  double maxLength = 0;
  long int voxelIndex = 0;
  unsigned long int parameterIndex = 0;
  unsigned int dimensionIndex = 0;
  
  parameterIndex = 0;
  for (voxelIndex = 0; voxelIndex < numberOfGridVoxels; voxelIndex++)
    {
      length = 0;
      for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
        {
          length += (next.GetElement(parameterIndex) * next.GetElement(parameterIndex));
          parameterIndex++;
        }
      length = sqrt(length);

      if (length > maxLength)
        {
          maxLength = length;
        }  
    }
  niftkitkDebugMacro(<< "LineAscent():Max length of all gradient vectors =" << maxLength << ", minimum threshold=" << this->GetMinimumGradientVectorMagnitudeThreshold());

  // Now do line minimisation.
  bool improvement = false;
  double nextValue = 0;  
  double bestValue = this->m_Value;
  ParametersType localBestParameters = current;
  ParametersType localNextParameters = current;

  while(this->GetStepSize() > this->GetMinimumStepSize()){
  
    double scalingFactor = this->GetStepSize()/maxLength;

    niftkitkDebugMacro(<< "LineAscent():scalingFactor=" << scalingFactor \
        << ", maxDisplacement=" << this->GetStepSize() \
        << ", maxGradValue=" << maxLength \
        << ", minStepSize=" << this->GetMinimumStepSize());

    voxelIndex = 0;        
    parameterIndex = 0;
      
    for (voxelIndex = 0; voxelIndex < numberOfGridVoxels; voxelIndex++)
      {
        for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
          {
            localNextParameters.SetElement(parameterIndex, localBestParameters[parameterIndex] - (next.GetElement(parameterIndex)*scalingFactor));
            parameterIndex++;
          }              
        //printf("Matt:%d,%f,%f,%f\n",voxelIndex, localNextParameters.GetElement(parameterIndex-3), localNextParameters.GetElement(parameterIndex-2), localNextParameters.GetElement(parameterIndex-1));
      }

    nextValue = this->GetCostFunction()->GetValue(localNextParameters);
    niftkitkDebugMacro(<< "LineAscent():newSimilarity=" << nextValue);

    // Check if its any better
    if (  (this->m_Maximize && nextValue > bestValue)
       || (!this->m_Maximize && nextValue < bestValue))
      {
        niftkitkDebugMacro(<< "LineAscent():nextValue:" << nextValue << ", is better than bestValue:" << bestValue);
        bestValue = nextValue;
        localBestParameters = localNextParameters;
        improvement = true;
      }
    else
      {              
        this->SetStepSize(this->GetStepSize() * this->m_IteratingStepSizeReductionFactor);

        niftkitkDebugMacro(<< "LineAscent():nextValue:" << nextValue \
            << ", is worse than bestValue:" << bestValue \
            << ", so reducing step size by:" << this->m_IteratingStepSizeReductionFactor \
            << ", to:" << this->GetStepSize());
      }

  }
  next = localBestParameters;
  
  if(improvement){
    
    double maxStep = 0;
    parameterIndex=0;
    
    for (voxelIndex = 0; voxelIndex < numberOfGridVoxels; voxelIndex++)
      {
        length = 0;
        for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
          {
            length += (
                        (current.GetElement(parameterIndex) - next.GetElement(parameterIndex))
                        *
                        (current.GetElement(parameterIndex) - next.GetElement(parameterIndex))
                        );
            parameterIndex++;
          }
        length = sqrt(length);

        if (length > maxStep)
          {
            maxStep = length;
          }  
      }
    niftkitkDebugMacro(<< "LineAscent():Setting m_StepSize to:" << maxStep << ", min step size stays at:" << this->GetMinimumStepSize());
    this->SetStepSize(maxStep);
  }
  if(!improvement){
    niftkitkInfoMacro(<< "LineAscent():No Further metric improvement");
  }
  else{
    niftkitkInfoMacro(<< "LineAscent():[" << iterationNumber << "] New metric value: " << bestValue);
  }
  
  niftkitkDebugMacro(<< "LineAscent():Finished");
  return improvement;
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
double
FFDGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::CalculateNextStep(int iterationNumber, double currentSimilarity, const ParametersType& current, ParametersType& next)
{
  niftkitkDebugMacro(<< "CalculateNextStep():Started");
  
  unsigned long int numberOfGridVoxels = 1;
  BSplineTransformPointer transform = dynamic_cast<BSplineTransformPointer>(this->m_DeformableTransform.GetPointer());
  if (transform == 0)
    {
      niftkitkExceptionMacro(<< "Can't dynamic cast to BSplineTransform");
    }
  GridImagePointer gridPointer = transform->GetGrid();
  OutputImageSizeType size = gridPointer->GetLargestPossibleRegion().GetSize();
  for (int i = 0; i < Dimension; i++)
    {
      numberOfGridVoxels *= size[i];
    }  
  
  this->OptimizeNextStep(iterationNumber, numberOfGridVoxels, current, next);

  this->m_CalculateNextStepCounter++;
  niftkitkDebugMacro(<< "CalculateNextStep():Finished");
  
  return std::numeric_limits<double>::max(); 
}

} // namespace itk.

#endif
