/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter_txx
#define __itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter_txx

#include "itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionIterator.h>

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NDimensions>
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::FourthOrderRungeKuttaVelocityFieldIntegrationFilter()
{
  m_TimeVaryingVelocityFieldInterpolator = TimeVaryingVelocityFieldInterpolatorType::New();
  m_MaxDistanceImageInterpolator = ThicknessImageInterpolatorType::New();
  
  m_MaxDistanceMask = ThicknessImageType::New();
  m_MaxDistanceMask = NULL;

  m_MaskImage = MaskImageType::New();
  m_MaskImage = NULL;

  m_GreyWhiteInterface = MaskImageType::New();
  m_GreyWhiteInterface = NULL;
  
  m_ThicknessImage = ThicknessImageType::New();
  m_HitsImage = ThicknessImageType::New();

  m_StartTime = 0;
  m_FinishTime = 1;
  m_DeltaTime = 0.1;
  m_MaxThickness = 0;
  m_MaxDisplacement = 0;
  m_FieldEnergy = 0;
  
  m_CalculateThickness = false;
}

template <class TScalarType, unsigned int NDimensions>
void
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "DeltaTime=" << m_DeltaTime << std::endl;
  os << indent << "StartTime=" << m_StartTime << std::endl;
  os << indent << "FinishTime=" << m_FinishTime << std::endl;
  os << indent << "CalculateThickness=" << m_CalculateThickness << std::endl;
  os << indent << "MaxThickness=" << m_MaxThickness << std::endl;
  os << indent << "MaxDisplacement=" << m_MaxDisplacement << std::endl;
  os << indent << "FieldEnergy=" << m_FieldEnergy << std::endl;
}

template <class TScalarType, unsigned int NDimensions>
void 
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::GenerateOutputInformation()
{
  typename Superclass::OutputImagePointer      outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer  inputPtr  = this->GetInput();

  if ( !outputPtr || !inputPtr)
    {
    return;
    }

  TimeVaryingVelocityRegionType inputRegion = inputPtr->GetLargestPossibleRegion();
  TimeVaryingVelocityIndexType inputIndex = inputRegion.GetIndex();
  TimeVaryingVelocitySizeType inputSize = inputRegion.GetSize();
  TimeVaryingVelocityPointType inputOrigin = inputPtr->GetOrigin();
  TimeVaryingVelocitySpacingType inputSpacing = inputPtr->GetSpacing();
  TimeVaryingVelocityDirectionType inputDirection = inputPtr->GetDirection();
  
  DisplacementImageIndexType outputIndex;
  DisplacementImageSizeType outputSize;
  DisplacementImagePointType outputOrigin;
  DisplacementImageSpacingType outputSpacing;
  DisplacementImageDirectionType outputDirection;
  
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      outputIndex[i] = inputIndex[i];
      outputSize[i] = inputSize[i];
      outputOrigin[i] = inputOrigin[i];
      outputSpacing[i] = inputSpacing[i];
      
      for (unsigned int j = 0; j < NDimensions; j++)
        {
          outputDirection[i][j] = inputDirection[i][j];
        }
    }
  DisplacementImageRegionType outputRegion;
  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);
  
  outputPtr->SetRegions(outputRegion);
  outputPtr->SetOrigin(outputOrigin);
  outputPtr->SetSpacing(outputSpacing);
  outputPtr->SetDirection(outputDirection);
  outputPtr->SetNumberOfComponentsPerPixel(inputPtr->GetNumberOfComponentsPerPixel());
}

template <class TScalarType, unsigned int NDimensions>
void 
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::BeforeThreadedGenerateData()
{
  if (m_CalculateThickness)
    {
      typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
      DisplacementImageSizeType outputSize = outputPtr->GetLargestPossibleRegion().GetSize();
      DisplacementImageSizeType thicknessSize = m_ThicknessImage->GetLargestPossibleRegion().GetSize();
      
      if (outputSize != thicknessSize)
        {
          DisplacementImageRegionType outputRegion = outputPtr->GetLargestPossibleRegion();
          DisplacementImagePointType outputOrigin = outputPtr->GetOrigin();
          DisplacementImageSpacingType outputSpacing = outputPtr->GetSpacing();
          DisplacementImageDirectionType outputDirection = outputPtr->GetDirection();

          m_ThicknessImage->SetRegions(outputRegion);
          m_ThicknessImage->SetOrigin(outputOrigin);
          m_ThicknessImage->SetSpacing(outputSpacing);
          m_ThicknessImage->SetDirection(outputDirection);
          m_ThicknessImage->Allocate();
          
          m_HitsImage->SetRegions(outputRegion);
          m_HitsImage->SetOrigin(outputOrigin);
          m_HitsImage->SetSpacing(outputSpacing);
          m_HitsImage->SetDirection(outputDirection);
          m_HitsImage->Allocate();
          
        }      
    }
}

template <class TScalarType, unsigned int NDimensions>
void 
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::AfterThreadedGenerateData()
{
  if (m_CalculateThickness)
    {
      m_ThicknessImage->FillBuffer(0);
      m_HitsImage->FillBuffer(0);
      m_MaxThickness = 0;
      
      // After the filter has run, we know the output image has the integrated displacement vectors.
      // Hence we have the thickness values (magnitude of displacement vector) directly from output image.
      // So, now we want to re-run the integration, using the thickness value, and propogating it through the masked region.

      this->IntegrateRegion(m_ThicknessImage->GetLargestPossibleRegion(), 1.0, false, true);
     
      // And we divide the thickness image by the hits image.
      // See Das et. al. NeuroImage 2009, page 871, section Thickness Propogation.
      ImageRegionIterator<ThicknessImageType> thicknessIterator(m_ThicknessImage, m_ThicknessImage->GetLargestPossibleRegion());      
      ImageRegionConstIterator<ThicknessImageType> hitsIterator(m_HitsImage, m_HitsImage->GetLargestPossibleRegion());
      
      for (hitsIterator.GoToBegin(),
          thicknessIterator.GoToBegin();
          !hitsIterator.IsAtEnd();
          ++hitsIterator,
          ++thicknessIterator)
        {
          if (hitsIterator.Get() > 0)
            {
              thicknessIterator.Set(thicknessIterator.Get()/(double)hitsIterator.Get());
              
              if (thicknessIterator.Get() > m_MaxThickness)
              {
                m_MaxThickness = thicknessIterator.Get();
              }
            }
          else
            {
              thicknessIterator.Set(0);
            }
        }
    }
    
    double norm;    
    m_MaxDisplacement = 0;
    m_FieldEnergy = 0;
    
    typename DisplacementImageType::Pointer outputImage 
    = static_cast< DisplacementImageType * >(this->ProcessObject::GetOutput(0));
    
    ImageRegionConstIterator<DisplacementImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
    for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
    {
      norm = outputIterator.Get().GetNorm();
      m_FieldEnergy += norm;
      
      if (norm > m_MaxDisplacement)
      {
        m_MaxDisplacement = norm;
      }
    }
    
    niftkitkDebugMacro(<<"AfterThreadedGenerateData():MaxDisplacement=" << m_MaxDisplacement \
      << ", m_MaxThickness=" << m_MaxThickness \
      << ", m_FieldEnergy=" << m_FieldEnergy \
      );
}

template <class TScalarType, unsigned int NDimensions>
void
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::ThreadedGenerateData(const DisplacementImageRegionType& regionForThread, ThreadIdType threadNumber)
{
  niftkitkDebugMacro(<<"ThreadedGenerateData():Started thread:" << threadNumber);

  this->IntegrateRegion(regionForThread, 1.0, true, false);
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():Finished thread:" << threadNumber);
}

template <class TScalarType, unsigned int NDimensions>
void 
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::IntegrateRegion(const DisplacementImageRegionType &regionForThread, float directionOverride, bool writeToOutput, bool writeToThicknessImage)
{
  typename TimeVaryingVelocityImageType::Pointer inputImage 
    = static_cast< TimeVaryingVelocityImageType * >(this->ProcessObject::GetInput(0));

  typename DisplacementImageType::Pointer outputImage 
    = static_cast< DisplacementImageType * >(this->ProcessObject::GetOutput(0));

  float startTime = m_StartTime;
  float finishTime = m_FinishTime;
  
  if (startTime < 0) 
    {
      startTime = 0;
    }
  if (startTime > 1) 
    {
      startTime = 1;
    }
  if (finishTime < 0) 
    {
      finishTime = 0;
    }
  if (finishTime > 1) 
    {
      finishTime = 1;
    }

  float timeDirection = 1;
  if(startTime > finishTime)
    {
      timeDirection = -1;
    }
  timeDirection *= directionOverride;
  int numberOfTimePoints = inputImage->GetLargestPossibleRegion().GetSize()[Dimension];
  
  niftkitkDebugMacro(<<"ThreadedGenerateData():m_StartTime=" << m_StartTime \
    << ", m_FinishTime=" << m_FinishTime \
    << ", startTime=" << startTime \
    << ", finishTime=" << finishTime \
    << ", timeDirection=" << timeDirection \
    << ", writeToOutput=" << writeToOutput \
    << ", writeToThicknessImage=" << writeToThicknessImage \
    << ", numberOfTimePoints=" << numberOfTimePoints \
    );

  DisplacementPixelType zero;
  zero.Fill(0);
  
  ImageRegionIteratorWithIndex<DisplacementImageType> outputIterator(outputImage, regionForThread);
  
  if (writeToOutput)
    {
      for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
        {
          outputIterator.Set(zero);
        }      
    }
  
  if (startTime == finishTime)
    {
	  niftkitkDebugMacro(<<"ThreadedGenerateData():startTime == finishTime, so returning zero displacement");
      return;
    }

  DisplacementImageIndexType index;
  DisplacementPixelType disp;
  bool doIntegration;
  
  m_TimeVaryingVelocityFieldInterpolator->SetInputImage(inputImage);

  for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
    {
      doIntegration = true;
      index = outputIterator.GetIndex();
      
      if (writeToThicknessImage)
        {
          // So, we integrate pixels where the m_GreyWhiteInterface != 0.
          if (m_GreyWhiteInterface.IsNotNull() && m_GreyWhiteInterface->GetPixel(index) == 0)
            {
              doIntegration = false;
            }
        }
      else
        {
          // So, we only integrate pixels where m_MaskImage != 0.
          if (m_MaskImage.IsNotNull() && m_MaskImage->GetPixel(index) == 0)
            {
              doIntegration = false;
            }          
        }
      
      if (doIntegration)
        {
          this->IntegratePoint(startTime, finishTime, timeDirection, numberOfTimePoints, index, inputImage, writeToThicknessImage, disp);
        }
      else
        {
          disp = zero;
        }
      
      if (writeToOutput)
        {
          outputIterator.Set(disp);          
        }
    }
    
}

template <class TScalarType, unsigned int NDimensions>
void
FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, NDimensions>
::IntegratePoint(const float& startTime, 
                 const float& endTime, 
                 const float& timeDirection, 
                 const unsigned int& numberOfTimePoints,
                 const DisplacementImageIndexType& index,
                 const TimeVaryingVelocityImageType* velocityField,
                 const bool& writeToThicknessImage,
                 DisplacementPixelType& displacement)
{

  // This is the output point, i.e. the result, which we initialize to zero.
  displacement.Fill(0);

  // If start and end time are the same, we don't need to integrate, so return.
  if (startTime == endTime)
    {
      return;
    }
  
  // This is the initial point, where we start the integrating from.
  // i.e. the millimetre point that corresponds to the index, and hence is static.
  TimeVaryingPointType initialPoint; 
  initialPoint.Fill(0);
  
  // This is the current point, as we go through the integration.
  // So in Runge-Kutta terms it is the first of the four points we integrate.
  TimeVaryingPointType currentPoint; 
  currentPoint.Fill(0);
  
  // This is the next point, so after you have calculated the 4 Runge-Kutta points
  // this is the point you update to, i.e. it is the next position.
  TimeVaryingPointType nextPoint; 
  nextPoint.Fill(0);
  
  // These are the 4 values we use for 4th order Runge-Kutta integration
  TimeVaryingPointType p1; 
  TimeVaryingPointType p2;
  TimeVaryingPointType p3;
  TimeVaryingPointType p4;
  p1.Fill(0);
  p2.Fill(0);
  p3.Fill(0);
  p4.Fill(0);
  
  // These are the function values at these 4 points.
  // Also, note that we are calculating a displacement, by integrating
  // a velocity field. A velocity field v = ds/dt IS the gradient.
  TimeVaryingVelocityPixelType f1; 
  TimeVaryingVelocityPixelType f2; 
  TimeVaryingVelocityPixelType f3; 
  TimeVaryingVelocityPixelType f4; 
  f1.Fill(0);
  f2.Fill(0);
  f3.Fill(0);
  f4.Fill(0);
  
  // These are the k1-k4 values mentioned in Numerical Recipes.
  TimeVaryingPointType k1; 
  TimeVaryingPointType k2;
  TimeVaryingPointType k3;
  TimeVaryingPointType k4;
  k1.Fill(0);
  k2.Fill(0);
  k3.Fill(0);
  k4.Fill(0);

  // If we are doing thickness calculations, we need a point to pass to interpolator, and an index.
  DisplacementImagePointType thicknessPoint;
  DisplacementImageIndexType thicknessIndex;
  DisplacementPixelType displacementValue;
  ThicknessImagePixelType totalDisplacementValue;
  int hitsImageCounter;
  
  // We start by converting the input index to millimetres,
  // so integration is in millimetre space. We need to convert it
  // because the input index is specified in the displacement image (3D),
  // but the TimeVaryingVelocityIndexType is 4D.
  
  TimeVaryingVelocityIndexType velocityIndex;
  velocityIndex.Fill(0);
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      velocityIndex[i] = index[i];
    }
  
  velocityField->TransformIndexToPhysicalPoint(velocityIndex, initialPoint);
  initialPoint[NDimensions]= startTime*(numberOfTimePoints-1);
  
  bool isTooThick = false;
  bool timeDone = false;
  float tmp = 0;
  float magnitudeOfThisDisplacement = 0;
  float magnitudeOfTotalDisplacement = 0;
  double integratingTimePlusH = 0;
  double integratingTimePlusHalfH = 0;

  double deltaTime = m_DeltaTime;
  float numberOfTimePointsMinusOne = numberOfTimePoints - 1;

  double timeStep = timeDirection*deltaTime;  
  float integratingTime = startTime;

  while(!timeDone)
    {
      // displacement[i] is initialized to zero, and at each iteration it grows.
      // p1 - p4 are the 4 points necessary for Runge-Kutta.
      
      for (unsigned int i = 0; i < Dimension; i++)
        {
          currentPoint[i] = initialPoint[i] + displacement[i]; 
          p1[i] = currentPoint[i];
          p2[i] = currentPoint[i];
          p3[i] = currentPoint[i];
          p4[i] = currentPoint[i];
          
          // Leave this in here, just in case, the nextPoint isnt updated, due to 
          // the point running out of bounds, and hence the update being excluded in
          // the chain of nested if statements below.
          nextPoint[i] = currentPoint[i];                             
        }
      
      integratingTimePlusH = integratingTime + timeStep;
      if (integratingTimePlusH < 0) integratingTimePlusH = 0;
      if (integratingTimePlusH > 1) integratingTimePlusH = 1;

      integratingTimePlusHalfH = integratingTime + timeStep*0.5;
      if (integratingTimePlusHalfH < 0) integratingTimePlusHalfH = 0;
      if (integratingTimePlusHalfH > 1) integratingTimePlusHalfH = 1;

      // So, as integratingTime increases, the time coordinate of p1 - p4
      // will index through the time varying velocity field.
      // So, the 4th ordinate (time ordinate) is a fraction from 0-1.
      // (or more precisely, startTime -> endTime, which should be 0,1 or 1,0).
      p1[NDimensions] = integratingTime          * numberOfTimePointsMinusOne;
      p2[NDimensions] = integratingTimePlusHalfH * numberOfTimePointsMinusOne;
      p3[NDimensions] = integratingTimePlusHalfH * numberOfTimePointsMinusOne;
      p4[NDimensions] = integratingTimePlusH     * numberOfTimePointsMinusOne;

      f1.Fill(0);
      f2.Fill(0);
      f3.Fill(0);
      f4.Fill(0);
      
      // If we compare this code with Numerical Recipes in C, page 711,
      // the bit under fourth-order Runge-Kutta formula, 
      // then we are integrating over time, so our time dimension (dimension 3),
      // it equivalent to x_n in the book.
      
      // Also, note that we are calculating a displacement, by integrating
      // a velocity field. A velocity field v = ds/dt IS the gradient.
      
      if (this->m_TimeVaryingVelocityFieldInterpolator->IsInsideBuffer(p1))
        {
          f1 = this->m_TimeVaryingVelocityFieldInterpolator->Evaluate(p1);
          for (unsigned int i = 0; i < Dimension; i++)
            {
              k1[i] = timeStep * f1[i];
              p2[i] += (k1[i] * 0.5);
            }

          if (this->m_TimeVaryingVelocityFieldInterpolator->IsInsideBuffer(p2) )
            {
              f2 = this->m_TimeVaryingVelocityFieldInterpolator->Evaluate(p2);
              for (unsigned int i = 0; i < Dimension; i++)
                {
                  k2[i] = timeStep * f2[i];
                  p3[i] += (k2[i] * 0.5);
                }
                            
              if (this->m_TimeVaryingVelocityFieldInterpolator->IsInsideBuffer(p3) )
                {
                  f3 = this->m_TimeVaryingVelocityFieldInterpolator->Evaluate(p3);
                  for (unsigned int i = 0; i < Dimension; i++)
                    {
                      k3[i] = timeStep * f3[i];
                      p4 += k3[i];
                    }

                  if (this->m_TimeVaryingVelocityFieldInterpolator->IsInsideBuffer(p4) )
                    {
                      f4 = this->m_TimeVaryingVelocityFieldInterpolator->Evaluate(p4);
                      for (unsigned int i = 0; i < Dimension; i++)
                        {
                          k4[i] = timeStep * f3[i];
                        }
                    }

                  for (unsigned int i = 0; i < NDimensions; i++)
                    {
                      nextPoint[i] = currentPoint[i] + k1[i]/6.0 + k2[i]/3.0 + k3[i]/3.0 + k4[i]/6.0;
                    }
                  nextPoint[Dimension]= integratingTime * numberOfTimePointsMinusOne;

                  for (unsigned int i = 0; i < NDimensions; i++)
                    {
                      displacement[i] = nextPoint[i] - initialPoint[i];                      
                    }
                  
                  if (writeToThicknessImage)
                    {
                      displacementValue = this->GetOutput()->GetPixel(index);
                          
                      for (unsigned int i = 0; i < NDimensions; i++)
                        {
                          thicknessPoint[i] = nextPoint[i];
                        }
                          
                      m_ThicknessImage->TransformPhysicalPointToIndex(thicknessPoint, thicknessIndex);
                          
                      totalDisplacementValue = m_ThicknessImage->GetPixel(thicknessIndex) + displacementValue.GetNorm();
                      m_ThicknessImage->SetPixel(thicknessIndex, totalDisplacementValue);
                          
                      hitsImageCounter = (int)(m_HitsImage->GetPixel(thicknessIndex)) + 1;
                      m_HitsImage->SetPixel(thicknessIndex, hitsImageCounter);
                      
                    }

                  if (m_MaxDistanceMask.IsNotNull())
                    {
                      magnitudeOfTotalDisplacement = 0;  
                                          
                      for (unsigned int i = 0; i < NDimensions; i++)
                        {
                          thicknessPoint[i] = nextPoint[i];
                          
                          tmp = (nextPoint[i] - initialPoint[i]);
                          magnitudeOfTotalDisplacement += (tmp*tmp);                          
                        }
                        
                      magnitudeOfTotalDisplacement = sqrt(magnitudeOfTotalDisplacement);
                         
                      m_MaxDistanceImageInterpolator->SetInputImage(m_MaxDistanceMask);
                      
                      if (!m_MaxDistanceImageInterpolator->IsInsideBuffer(thicknessPoint)
                          || (m_MaxDistanceImageInterpolator->IsInsideBuffer(thicknessPoint) && 
                              magnitudeOfTotalDisplacement > m_MaxDistanceImageInterpolator->Evaluate(thicknessPoint)))
                        {
                          isTooThick = true;
                        }                      
                    }
               
                } // end if y3 still inside buffer
              else
                {
                  timeDone = true;
                }
            } // end if y2 still inside buffer
          else
            {
              timeDone = true;
            }
        } // end if y1 still inside buffer
      else
        {
          timeDone = true;
        }
        
      magnitudeOfThisDisplacement = 0;
      
      for (unsigned int i = 0; i < NDimensions; i++)
        {
          tmp = (nextPoint[i] - currentPoint[i]);
          magnitudeOfThisDisplacement += (tmp*tmp);
        }                  

      integratingTime = integratingTime + deltaTime*timeDirection;
      
      if (startTime > endTime)
        {
          if (integratingTime <= endTime)
            {
              timeDone = true;  
            }
        } 
      else
        {
          if (integratingTime >= endTime)
            {
              timeDone=true;
            }
        } 

      if (magnitudeOfThisDisplacement == 0)
        {
          timeDone = true;    
        }
      else if (isTooThick)
        {
          timeDone = true;    
        }
      
    } // end while ! time done.
    
} // end function

} // end namespace

#endif
