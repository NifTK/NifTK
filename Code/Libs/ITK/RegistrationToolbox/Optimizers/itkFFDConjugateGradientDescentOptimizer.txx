/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKFFDConjugateGradientDescentOptimizer_TXX_
#define ITKFFDConjugateGradientDescentOptimizer_TXX_

#include "itkFFDConjugateGradientDescentOptimizer.h"
#include "itkLogHelper.h"

namespace itk
{
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
FFDConjugateGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::FFDConjugateGradientDescentOptimizer()
{
  this->conjugateG = NULL;
  this->conjugateH = NULL;
}

/*
 * PrintSelf
 */
template < typename TFixedImage, typename TMovingImage, class TScalarType, class TDeformationScalar >
void
FFDConjugateGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDConjugateGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::Initialize()
{
  niftkitkDebugMacro(<< "Initialize():Started");
  
  this->m_NumberOfGridVoxels = 1;
  BSplineTransformPointer transform = dynamic_cast<BSplineTransformPointer>(this->m_DeformableTransform.GetPointer());
  if (transform == 0)
    {
      itkExceptionMacro(<< "Can't dynamic cast to BSplineTransform");
    }
  GridImagePointer gridPointer = transform->GetGrid();
  OutputImageSizeType size = gridPointer->GetLargestPossibleRegion().GetSize();
  for (int i = 0; i < Dimension; i++)
    {
      m_NumberOfGridVoxels *= size[i];
    }  
  
  if (this->conjugateG != NULL)
    {
      niftkitkDebugMacro(<< "Initialize():Deleting old conjugateG");
      delete[] this->conjugateG;
    }

  this->conjugateG = new float3[m_NumberOfGridVoxels];

  niftkitkDebugMacro(<< "Initialize():set conjugateG to " << m_NumberOfGridVoxels << " voxels");
  
  if (this->conjugateH != NULL)
    {
      niftkitkDebugMacro(<< "Initialize():Deleting old conjugateH");
      delete[] this->conjugateH;  
    }  
  
  this->conjugateH = new float3[m_NumberOfGridVoxels];    
  
  niftkitkDebugMacro(<< "Initialize():set conjugateH to " << m_NumberOfGridVoxels << " voxels");
  
  niftkitkDebugMacro(<< "Initialize():Finished");
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDConjugateGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::StoreGradient(const ParametersType& gradient)
{
  niftkitkDebugMacro(<< "StoreGradient():Started");
  
  unsigned long int parameterIndex=0;
  for(unsigned int i=0; i<m_NumberOfGridVoxels; i++){
    this->conjugateH[i].x = this->conjugateG[i].x = -gradient.GetElement(parameterIndex++);
    this->conjugateH[i].y = this->conjugateG[i].y = -gradient.GetElement(parameterIndex++);
    this->conjugateH[i].z = this->conjugateG[i].z = -gradient.GetElement(parameterIndex++);
  }    
  
  niftkitkDebugMacro(<< "StoreGradient():Finished");
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDConjugateGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::GetNextGradient(const ParametersType& currentGradient, ParametersType& nextGradient)
{
  niftkitkDebugMacro(<< "GetNextGradient():Started");
  
  float dgg=0.0f, gg=0.0f;
  
  unsigned long int parameterIndex = 0;
  
  for(unsigned int i=0; i<m_NumberOfGridVoxels; i++){
    gg += this->conjugateG[i].x * this->conjugateG[i].x;
    gg += this->conjugateG[i].y * this->conjugateG[i].y;
    gg += this->conjugateG[i].z * this->conjugateG[i].z;
    
    dgg += (currentGradient.GetElement(parameterIndex) + this->conjugateG[i].x) * currentGradient.GetElement(parameterIndex);
    parameterIndex++;
    dgg += (currentGradient.GetElement(parameterIndex) + this->conjugateG[i].y) * currentGradient.GetElement(parameterIndex);
    parameterIndex++;
    dgg += (currentGradient.GetElement(parameterIndex) + this->conjugateG[i].z) * currentGradient.GetElement(parameterIndex);
    parameterIndex++;
  }
  float gam=dgg/gg;
  parameterIndex = 0;
  for(unsigned int i=0; i<m_NumberOfGridVoxels; i++){
    
    this->conjugateG[i].x = -currentGradient.GetElement(parameterIndex);
    this->conjugateH[i].x = this->conjugateG[i].x + gam*this->conjugateH[i].x;
    nextGradient.SetElement(parameterIndex, -this->conjugateH[i].x);
    parameterIndex++;
    
    this->conjugateG[i].y = -currentGradient.GetElement(parameterIndex);
    this->conjugateH[i].y = this->conjugateG[i].y + gam*this->conjugateH[i].y;
    nextGradient.SetElement(parameterIndex, -this->conjugateH[i].y);
    parameterIndex++;
    
    this->conjugateG[i].z = -currentGradient.GetElement(parameterIndex);
    this->conjugateH[i].z = this->conjugateG[i].z + gam*this->conjugateH[i].z;
    nextGradient.SetElement(parameterIndex, -this->conjugateH[i].z);
    parameterIndex++;
  }
  
  niftkitkDebugMacro(<< "GetNextGradient():Finished");
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDConjugateGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::CleanUp()
{
  
  niftkitkDebugMacro(<< "CleanUp():Started");
  
  if (this->conjugateG != NULL)
    {
      niftkitkDebugMacro(<< "CleanUp():Deleting conjugateG");
      delete[] this->conjugateG;
      this->conjugateG = NULL;
    }
  if (this->conjugateH != NULL)
    {
      niftkitkDebugMacro(<< "CleanUp():Deleting conjugateH");
      delete[] this->conjugateH;  
      this->conjugateH = NULL;
    }
  
  niftkitkDebugMacro(<< "CleanUp():Finished");
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDConjugateGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::OptimizeNextStep(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next)
{
  niftkitkDebugMacro(<< "OptimizeNextStep():Started");
  
  if (m_DerivativeAtCurrentPosition.GetSize() != current.GetSize())
    {
      niftkitkDebugMacro(<< "OptimizeNextStep():Initializing derivative arrays");
      m_DerivativeAtCurrentPosition.SetSize(current.GetSize());
      m_DerivativeAtNextPosition.SetSize(current.GetSize());
      
      this->GetGradient(iterationNumber, current, m_DerivativeAtCurrentPosition);
      this->StoreGradient(m_DerivativeAtCurrentPosition);

    }

  next = m_DerivativeAtCurrentPosition;
  bool improvement = this->LineAscent(iterationNumber, numberOfGridVoxels, current, next);

  this->GetGradient(iterationNumber, next, m_DerivativeAtNextPosition);
  this->GetNextGradient(m_DerivativeAtNextPosition, m_DerivativeAtCurrentPosition);
  
  if (!improvement)
    {
      niftkitkDebugMacro(<< "OptimizeNextStep():No improvement found, setting step size to zero.");
      this->SetStepSize(0);
    }
  
  niftkitkDebugMacro(<< "OptimizeNextStep():Finished");
  
}

} // namespace itk.

#endif
