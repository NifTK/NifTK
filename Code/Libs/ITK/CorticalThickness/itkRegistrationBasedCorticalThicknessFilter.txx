/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRegistrationBasedCorticalThicknessFilter_txx
#define __itkRegistrationBasedCorticalThicknessFilter_txx

#include <niftkConversionUtils.h>
#include "itkRegistrationBasedCorticalThicknessFilter.h"
#include "itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter.h"
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkDemonsRegistrationFilter.h>
#include <itkDiffeomorphicDemonsRegistrationFilter.h>
#include "itkAddUpdateToTimeVaryingVelocityFieldFilter.h"
#include <itkGaussianSmoothVectorFieldFilter.h>
#include <itkWarpImageFilter.h>
#include <itkImageFileWriter.h>

#include <itkLogHelper.h>

namespace itk {

template< class TInputImage, typename TScalarType > 
RegistrationBasedCorticalThicknessFilter< TInputImage, TScalarType >
::RegistrationBasedCorticalThicknessFilter()
{
  m_MaxIterations = 100;
  m_M = 10;
  m_Epsilon = 0.001;
  m_UpdateSigma = 1.5;
  m_DeformationSigma = 0;
  m_Lambda = 1.0;
  m_Alpha = 0.99;
  m_FieldEnergy = 0;
  m_CostFunction = 0;
  m_MinJacobian = 1;
  m_MaxJacobian = 1;
  m_RMSChange = 0;
  m_MaxThickness = 0;
  m_MaxDisplacement = 0;
  m_SSD = 0;
  
  m_InterfaceDisplacementImage = VectorImageType::New();
  m_InterfaceDisplacementImage = NULL;
  
  niftkitkDebugMacro(<<"RegistrationBasedCTEFilter():Constructed with m_M=" << m_M \
      << ", m_Epsilon=" << m_Epsilon \
      << ", m_UpdateSigma=" << m_UpdateSigma \
      << ", m_DeformationSigma=" << m_DeformationSigma \
      << ", m_Lambda=" << m_Lambda \
      << ", m_Alpha=" << m_Alpha \
      << ", m_MaxIterations=" << m_MaxIterations \
      << ", m_FieldEnergy=" << m_FieldEnergy \
      << ", m_CostFunction=" << m_CostFunction \
      << ", m_MinJacobian=" << m_MinJacobian \
      << ", m_MaxJacobian=" << m_MaxJacobian \
      << ", m_RMSChange=" << m_RMSChange \
      << ", m_MaxThickness=" << m_MaxThickness \
      << ", m_MaxDisplacement=" << m_MaxDisplacement \
      << ", m_SSD=" << m_SSD \
      );
}

template< class TInputImage, typename TScalarType > 
void
RegistrationBasedCorticalThicknessFilter< TInputImage, TScalarType >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "M = " << m_M << std::endl;
  os << indent << "Epsilon = " << m_Epsilon << std::endl;
  os << indent << "UpdateSigma = " << m_UpdateSigma << std::endl;
  os << indent << "DeformationSigma = " << m_DeformationSigma << std::endl;
  os << indent << "Lambda = " << m_Lambda << std::endl;
  os << indent << "Alpha = " << m_Alpha << std::endl;
  os << indent << "MaxIterations = " << m_MaxIterations << std::endl;
  os << indent << "FieldEnergy = " << m_FieldEnergy << std::endl;
  os << indent << "CostFunction = " << m_CostFunction << std::endl;
  os << indent << "MinJacobian = " << m_MinJacobian << std::endl;
  os << indent << "MaxJacobian = " << m_MaxJacobian << std::endl;
  os << indent << "RMSChange = " << m_RMSChange << std::endl;
  os << indent << "MaxThickness = " << m_MaxThickness << std::endl;
  os << indent << "MaxDisplacement = " << m_MaxDisplacement << std::endl;
  os << indent << "SSD = " << m_SSD << std::endl;
}

template <typename TInputImage, typename TScalarType >   
void
RegistrationBasedCorticalThicknessFilter< TInputImage, TScalarType >
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");

  // Inputs are.....
  typename ImageType::Pointer whiteMatterPvMap = static_cast< ImageType * >(this->ProcessObject::GetInput(0));       
  typename ImageType::Pointer whitePlusGreyPvMap = static_cast< ImageType * >(this->ProcessObject::GetInput(1));       
  typename ImageType::Pointer thicknessPriorMap = static_cast< ImageType * >(this->ProcessObject::GetInput(2));
  typename MaskImageType::Pointer gwiBinaryImage = static_cast< MaskImageType * >(this->ProcessObject::GetInput(3));
  typename MaskImageType::Pointer greyMaskImage = static_cast< MaskImageType * >(this->ProcessObject::GetInput(4));
  
  // Make sure we have memory to write the output to.
  this->SetNumberOfOutputs(2);
  this->AllocateOutputs();

  // Output is final DiReCT map, an image of thickness values.
  typename ImageType::Pointer outputThicknessImage = static_cast< ImageType * >(this->ProcessObject::GetOutput(0));
  typename ImageType::Pointer outputTransformedMovingImage = static_cast< ImageType * >(this->ProcessObject::GetOutput(1));

  // Having multiple outputs doesn't work properly, the second output fails.
  if (outputTransformedMovingImage.IsNull())
    {
	  niftkitkDebugMacro(<<"GenerateData():Creating second output");
      outputTransformedMovingImage = ImageType::New();
      outputTransformedMovingImage->SetRegions(whiteMatterPvMap->GetLargestPossibleRegion());
      outputTransformedMovingImage->SetSpacing(whiteMatterPvMap->GetSpacing());
      outputTransformedMovingImage->SetOrigin(whiteMatterPvMap->GetOrigin());
      outputTransformedMovingImage->SetDirection(whiteMatterPvMap->GetDirection());
      outputTransformedMovingImage->Allocate();
      outputTransformedMovingImage->FillBuffer(0);
      this->SetNthOutput( 1, outputTransformedMovingImage.GetPointer() );      
    }
  
  if (m_InterfaceDisplacementImage.IsNull())
    {
      m_InterfaceDisplacementImage = VectorImageType::New();
      m_InterfaceDisplacementImage->SetRegions(whiteMatterPvMap->GetLargestPossibleRegion());
      m_InterfaceDisplacementImage->SetSpacing(whiteMatterPvMap->GetSpacing());
      m_InterfaceDisplacementImage->SetOrigin(whiteMatterPvMap->GetOrigin());
      m_InterfaceDisplacementImage->SetDirection(whiteMatterPvMap->GetDirection());
      m_InterfaceDisplacementImage->Allocate();
      
      VectorPixelType zero;
      zero.Fill(0);
      
      m_InterfaceDisplacementImage->FillBuffer(zero);
    }
  
  // Start. Check inputs all same size
  SizeType referenceSize = whiteMatterPvMap->GetLargestPossibleRegion().GetSize();
  SizeType checkSize;
  
  checkSize = whitePlusGreyPvMap->GetLargestPossibleRegion().GetSize();
  
  if (referenceSize != checkSize)
  {
    itkExceptionMacro(<< "White matter image size " << referenceSize \
      << ", is not the same as the white plus grey image size = " << checkSize);
  }
  
  checkSize = whitePlusGreyPvMap->GetLargestPossibleRegion().GetSize();
  
  if (referenceSize != checkSize)
  {
    itkExceptionMacro(<< "White matter image size " << referenceSize \
      << ", is not the same as the thickness prior image size = " << checkSize);
  }
  
  checkSize = gwiBinaryImage->GetLargestPossibleRegion().GetSize();
  
  if (referenceSize != checkSize)
  {
    itkExceptionMacro(<< "White matter image size " << referenceSize \
      << ", is not the same as the grey white boundary image size = " << checkSize);
  }
  
  // Get info from wm image
  VectorImageSizeType size = whiteMatterPvMap->GetLargestPossibleRegion().GetSize();
  VectorImageIndexType index = whiteMatterPvMap->GetLargestPossibleRegion().GetIndex();
  VectorImageSpacingType spacing = whiteMatterPvMap->GetSpacing();
  VectorImagePointType origin = whiteMatterPvMap->GetOrigin();
  VectorImageDirectionType direction = whiteMatterPvMap->GetDirection();
  VectorImageRegionType region = whiteMatterPvMap->GetLargestPossibleRegion();
  
  VectorImagePointer vectorField = VectorImageType::New();
  vectorField->SetRegions(region);
  vectorField->SetSpacing(spacing);
  vectorField->SetOrigin(origin);
  vectorField->SetDirection(direction);
  vectorField->Allocate();
  VectorPixelType zeroVector;
  zeroVector.Fill(0);
  vectorField->FillBuffer(zeroVector);
  
  // Create time varying velocity field.
  // Time dimension will have identity in direction field.
  TimeVaryingVectorImageSizeType timeVaryingSize;
  TimeVaryingVectorImageIndexType timeVaryingIndex;
  TimeVaryingVectorImageSpacingType timeVaryingSpacing;
  TimeVaryingVectorImagePointType timeVaryingOrigin;
  TimeVaryingVectorImageDirectionType timeVaryingDirection;
  TimeVaryingVectorImageRegionType timeVaryingRegion;

  for (unsigned int i = 0; i < Dimension; i++)
  {
    timeVaryingSize[i] = size[i];
    timeVaryingIndex[i] = index[i];
    timeVaryingSpacing[i] = spacing[i];
    timeVaryingOrigin[i] = origin[i];
    
    for (unsigned int j = 0; j < Dimension; j++)
    {
      timeVaryingDirection[i][j] = direction[i][j];
    }
  }
  
  timeVaryingSize[Dimension] = 1;
  timeVaryingIndex[Dimension] = 0;
  timeVaryingSpacing[Dimension] = 1.0;
  timeVaryingOrigin[Dimension] = 0;
  for (unsigned int j = 0; j < Dimension; j++)
  {
    timeVaryingDirection[Dimension][j] = 0;
  }
  timeVaryingDirection[Dimension][Dimension] = 1;
  
  niftkitkDebugMacro(<<"GenerateData():Input image size=" << size \
    << ", index=" << index \
    << ", spacing=" << spacing \
    << ", origin=" << origin \
    << ", direction=\n" << direction \
    );

  niftkitkDebugMacro(<<"GenerateData():Time varying velocity image size=" << timeVaryingSize \
    << ", index=" << timeVaryingIndex \
    << ", spacing=" << timeVaryingSpacing \
    << ", origin=" << timeVaryingOrigin \
    << ", direction=\n" << timeVaryingDirection \
    );
    
  timeVaryingRegion.SetSize(timeVaryingSize);
  timeVaryingRegion.SetIndex(timeVaryingIndex);
  
  TimeVaryingVectorImagePointer timeVaryingVelocityField = TimeVaryingVectorImageType::New();
  timeVaryingVelocityField->SetRegions(timeVaryingRegion);
  timeVaryingVelocityField->SetSpacing(timeVaryingSpacing);
  timeVaryingVelocityField->SetOrigin(timeVaryingOrigin);
  timeVaryingVelocityField->SetDirection(timeVaryingDirection);
  timeVaryingVelocityField->Allocate();
  
  TimeVaryingVectorImagePixelType noTimeAtAll;
  noTimeAtAll.Fill(0);
  
  timeVaryingVelocityField->FillBuffer(noTimeAtAll);
  
  typedef FourthOrderRungeKuttaVelocityFieldIntegrationFilter<TScalarType, Dimension> IntegratorType;
  typedef DisplacementFieldJacobianDeterminantFilter<VectorImageType, TScalarType> JacobianFilterType;
  typedef MinimumMaximumImageCalculator<ImageType> MinMaxCalculatorType;
  typedef WarpImageFilter<ImageType, ImageType, VectorImageType> WarpImageFilterType;
  typedef DemonsRegistrationFilter<ImageType, ImageType, VectorImageType> UpdateFilterType;
  typedef AddUpdateToTimeVaryingVelocityFieldFilter<TScalarType, Dimension> AddUpdateFilterType;
  typedef GaussianSmoothVectorFieldFilter<TScalarType, Dimension+1, Dimension> SmoothDeformationFilterType;
 
  typename IntegratorType::Pointer integrationFilter = IntegratorType::New();    
  typename SmoothDeformationFilterType::SigmaType deformationSigma;
  typename WarpImageFilterType::Pointer warpImageFilter = WarpImageFilterType::New();    

  unsigned int currentIteration = 0;
  
  double previousCost = std::numeric_limits<double>::max();
  double epsilon = std::numeric_limits<double>::max();
  
  TimeVaryingVectorImagePointer previousVelocityField = TimeVaryingVectorImageType::New();
  TimeVaryingVectorImagePointer currentVelocityField = TimeVaryingVectorImageType::New();
  
  currentVelocityField = timeVaryingVelocityField;
    
  // Make sure we definitely stop.
  while (currentIteration < m_MaxIterations)
  {
    // I'm testing with a circular object.
    // When we calculate the update, the velocity vectors point inwards.
    // i.e. pointing from the outer GM surface, inwards towards the WM.
    // So, when we integrate from 0-1, displacement vectors point inwards.
    // This is what we want, so that the transformation is a pull back from WM to GM.
    
    // This filter implements step 3(a) in Das et. al. NeuroImage 2009
    // We integrate the velocity field over M steps, with DeltaTime set to 1/M.
	  // At this stage the velocity field is computed, and we use the mask to
	  // stop the integration advancing.  We don't need the GM/WM interface image
	  // to do the thickness propagation.
	  
	  integrationFilter->SetInput(currentVelocityField);
    integrationFilter->SetStartTime(0);
    integrationFilter->SetFinishTime(1);
    integrationFilter->SetMaxDistanceMaskImage(thicknessPriorMap);
    integrationFilter->SetDeltaTime(1.0/this->m_M);
    integrationFilter->SetCalculateThickness(false);
    integrationFilter->Modified();

    // Take the latest transformation, and warp the moving image.
    
    warpImageFilter->SetInput(whiteMatterPvMap);
    warpImageFilter->SetOutputParametersFromImage(whitePlusGreyPvMap);
    warpImageFilter->SetDeformationField(integrationFilter->GetOutput());
    warpImageFilter->Modified();

    // We use the itkDemonsRegistrationFilter to compute 1 update.
    // Also, we can make use of the smoothing (of the update) therein.
    
    typename UpdateFilterType::Pointer updateFilter = UpdateFilterType::New();
    updateFilter->SetNumberOfIterations(1);
    updateFilter->SetFixedImage(whitePlusGreyPvMap);
    updateFilter->SetMovingImage(warpImageFilter->GetOutput());
    updateFilter->SetInitialDeformationField(vectorField);
    updateFilter->SetIntensityDifferenceThreshold(0.001);
    updateFilter->SetUseImageSpacing(true);
    updateFilter->SetSmoothUpdateField(true);
    updateFilter->SetUpdateFieldStandardDeviations(m_UpdateSigma);
    updateFilter->SetSmoothDeformationField(false);
    updateFilter->SetUseMovingImageGradient(true);
    updateFilter->Modified();
    updateFilter->Update();
    
    // Calculate Min and Max Jacobian.
    
    typename JacobianFilterType::Pointer jacobianFilter = JacobianFilterType::New();
    jacobianFilter->SetInput(integrationFilter->GetOutput());
    jacobianFilter->SetUseImageSpacing(true);
    jacobianFilter->Modified(); 
    jacobianFilter->Update();
    
    typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
    minMaxCalculator->SetImage(jacobianFilter->GetOutput());
    minMaxCalculator->Modified();
    minMaxCalculator->Compute();
    
    // Calculate similarity. SSD, as in equation 2 in paper.
    // The reason we dont use itkSSDImageToImageMetric, is because that class
    // would have to try and transform the image again, using an IdentityTransform
    
    m_SSD = 0;
    double diff = 0;
    ImageRegionConstIterator<ImageType> fixedImageIterator(whitePlusGreyPvMap, whitePlusGreyPvMap->GetLargestPossibleRegion());
    ImageRegionConstIterator<ImageType> movingImageIterator(warpImageFilter->GetOutput(), warpImageFilter->GetOutput()->GetLargestPossibleRegion());
    for (fixedImageIterator.GoToBegin(),
         movingImageIterator.GoToBegin();
         !fixedImageIterator.IsAtEnd();
         ++fixedImageIterator,
         ++movingImageIterator)
      {
        diff = movingImageIterator.Get() - fixedImageIterator.Get();
        m_SSD += (diff*diff);
      }
      
    // Retrieve some stats.
    
    m_MaxThickness = integrationFilter->GetMaxThickness();
    m_MaxDisplacement = integrationFilter->GetMaxDisplacement();
    m_FieldEnergy = integrationFilter->GetFieldEnergy();
    m_MinJacobian = minMaxCalculator->GetMinimum();
    m_MaxJacobian = minMaxCalculator->GetMaximum();
    m_RMSChange = updateFilter->GetRMSChange();
    m_Cost = (1.0-m_Alpha)*m_FieldEnergy + m_Alpha*m_SSD;
    epsilon = fabs((previousCost-m_Cost)/previousCost);
    
    // Calculate if we are continuing

    if (m_Cost > previousCost)
    {
      niftkitkDebugMacro(<<"GenerateData():currentCost=" << m_Cost \
        << ", which is > previousCost=" << previousCost \
        << ", so finishing main loop" \
        );
      
      currentVelocityField = previousVelocityField;
      break;
    }
    
    if (currentIteration > 2 && epsilon < m_Epsilon)
    {
      niftkitkDebugMacro(<<"GenerateData():Epsilon=" << epsilon \
        << ", previousCost=" << previousCost \
        << ", currentCost=" << m_Cost \
        << ", and tolerance=" << m_Epsilon \
        << ", so it looks like we are going nowhere, so finishing main loop." \
        );
        
      break;
    }
    
    typename AddUpdateFilterType::Pointer addUpdateFilter = AddUpdateFilterType::New();
    addUpdateFilter->SetInput(currentVelocityField);
    addUpdateFilter->SetUpdateImage(updateFilter->GetOutput());
    addUpdateFilter->SetInPlace(false);
    addUpdateFilter->SetTimePoint(1);
    addUpdateFilter->SetOverWrite(false);
    addUpdateFilter->SetScaleFactor(m_Lambda);
    addUpdateFilter->Modified();
    addUpdateFilter->UpdateLargestPossibleRegion();
    
    deformationSigma.Fill(m_DeformationSigma);
    deformationSigma[Dimension] = 0;

    typename SmoothDeformationFilterType::Pointer smoothDeformationFilter = SmoothDeformationFilterType::New();    
    smoothDeformationFilter->SetInput(addUpdateFilter->GetOutput());
    smoothDeformationFilter->SetSigma(deformationSigma);
    smoothDeformationFilter->Modified();
    smoothDeformationFilter->Update();
 
    previousVelocityField = currentVelocityField;
    currentVelocityField = smoothDeformationFilter->GetOutput();
    currentVelocityField->DisconnectPipeline();
 
    currentIteration++;
    previousCost = m_Cost;
    
    niftkitkInfoMacro(<<"GenerateData():[" << currentIteration \
      << "/" << m_MaxIterations \
      << "], tol=[" << epsilon \
      << "/" << m_Epsilon \
      << "], minj=" << m_MinJacobian \
      << ", maxj=" << m_MaxJacobian \
      << ", maxD=" << m_MaxDisplacement \
      << ", rms=" << m_RMSChange \
      << ", ssd=" << m_SSD \
      << ", fe=" << m_FieldEnergy \
      << ", al=" << m_Alpha \
      << ", cost=" << m_Cost \
      );

  }
  
  // Copy final transformed moving image to output 1.
  
  warpImageFilter->SetInput(whiteMatterPvMap);
  warpImageFilter->SetOutputParametersFromImage(whitePlusGreyPvMap);
  warpImageFilter->SetDeformationField(integrationFilter->GetOutput());
  warpImageFilter->Modified();
  warpImageFilter->Update();
  
  ImageRegionConstIterator<ImageType> warpedOutputIterator(warpImageFilter->GetOutput(), warpImageFilter->GetOutput()->GetLargestPossibleRegion());
  ImageRegionIterator<ImageType> transformedMovingImageOutputIterator(outputTransformedMovingImage, outputTransformedMovingImage->GetLargestPossibleRegion());
  for (warpedOutputIterator.GoToBegin(), transformedMovingImageOutputIterator.GoToBegin(); !transformedMovingImageOutputIterator.IsAtEnd(); ++transformedMovingImageOutputIterator, ++warpedOutputIterator)
    {
      transformedMovingImageOutputIterator.Set(warpedOutputIterator.Get());
    }
  
  // Do final DiReCT thickness propogation.
  
  // We have to integrate the other way round.
  // So this does one full integration from WM surface outwards towards GM surface,
  // so that for each voxel in gwiBinaryImage mask, we calculate a displacement vector.
  // Then we calculate the Euclidean norm of that vector, and repeat the integration
  // by starting at a voxel on the gwiBinaryImage mask, stepping outwards, pushing
  // the thickness value along the way to propogate it through the mask.
  
  integrationFilter->SetInput(currentVelocityField);
  integrationFilter->SetStartTime(1);
  integrationFilter->SetFinishTime(0);
  integrationFilter->SetDeltaTime(1.0/(this->m_M));
  integrationFilter->SetMaxDistanceMaskImage(thicknessPriorMap);
  integrationFilter->SetGreyWhiteInterfaceMaskImage(gwiBinaryImage);
  integrationFilter->SetCalculateThickness(true);
  integrationFilter->Modified();
  integrationFilter->Update();

  m_MaxThickness = integrationFilter->GetMaxThickness();
  m_MaxDisplacement = integrationFilter->GetMaxDisplacement();
  m_FieldEnergy = integrationFilter->GetFieldEnergy();
  
  niftkitkInfoMacro(<<"GenerateData():After thickness, m_FieldEnergy=" << m_FieldEnergy \
		  << ", m_MaxDisplacement=" << m_MaxDisplacement \
		  << ", m_MaxThickness=" << m_MaxThickness \
		  );

  // Copy the DiReCT thickness map to output 0.
  if (greyMaskImage.IsNotNull())
    {
      ImageRegionConstIterator<ImageType> thicknessIterator(integrationFilter->GetCalculatedThicknessImage(), integrationFilter->GetCalculatedThicknessImage()->GetLargestPossibleRegion());
      ImageRegionConstIterator<ImageType> maskIterator(greyMaskImage, greyMaskImage->GetLargestPossibleRegion());
      ImageRegionIterator<ImageType> outputIterator(outputThicknessImage, outputThicknessImage->GetLargestPossibleRegion());
      for (thicknessIterator.GoToBegin(),
           maskIterator.GoToBegin(),
           outputIterator.GoToBegin(); 
           !outputIterator.IsAtEnd(); 
           ++outputIterator,
           ++maskIterator,
           ++thicknessIterator)
      {
        if (maskIterator.Get() == 0)
          {
            outputIterator.Set(0);
          }
        else
          {
            outputIterator.Set(thicknessIterator.Get());    
          }
      }      
    }
  else
    {
      ImageRegionConstIterator<ImageType> thicknessIterator(integrationFilter->GetCalculatedThicknessImage(), integrationFilter->GetCalculatedThicknessImage()->GetLargestPossibleRegion());
      ImageRegionIterator<ImageType> outputIterator(outputThicknessImage, outputThicknessImage->GetLargestPossibleRegion());
      for (thicknessIterator.GoToBegin(),
           outputIterator.GoToBegin(); 
           !outputIterator.IsAtEnd(); 
           ++outputIterator,
           ++thicknessIterator)
      {
        outputIterator.Set(thicknessIterator.Get());    
      }      
      
    }
  
  // Copy the Vector displacement image to output.
  ImageRegionConstIterator<VectorImageType> transformationIterator(integrationFilter->GetOutput(), integrationFilter->GetOutput()->GetLargestPossibleRegion());
  ImageRegionConstIterator<ImageType> interfaceIterator(gwiBinaryImage, gwiBinaryImage->GetLargestPossibleRegion());
  ImageRegionIterator<VectorImageType> displacementIterator(m_InterfaceDisplacementImage, m_InterfaceDisplacementImage->GetLargestPossibleRegion());
  for (transformationIterator.GoToBegin(),
       displacementIterator.GoToBegin(),
       interfaceIterator.GoToBegin();
       !transformationIterator.IsAtEnd();
       ++transformationIterator,
       ++interfaceIterator,
       ++displacementIterator)
    {
      if (interfaceIterator.Get() > 0)
        {
          displacementIterator.Set(transformationIterator.Get());
        }
    }
       
  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif
