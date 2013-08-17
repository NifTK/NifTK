/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRegistrationBasedCTEFilter_txx
#define __itkRegistrationBasedCTEFilter_txx

#include <stdio.h>
#include "itkRegistrationBasedCTEFilter.h"
#include <itkContinuousIndex.h>
#include <itkPoint.h>
#include <itkImageFileWriter.h>

#include <itkLogHelper.h>

namespace itk {

template< class TInputImage, typename TScalarType > 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::RegistrationBasedCTEFilter()
{
  m_FinalPhiImage = VectorImageType::New();
  
  m_InitializePhiZeroFilter = InitializePhiFilterType::New();
  m_InitializePhiZeroFilter->SetInPlace(false);
  
  m_SmoothWMPVMapFilter = GaussianSmoothImageFilterType::New();
  m_SmoothGMWMPVMapFilter = GaussianSmoothImageFilterType::New();
  
  m_M = 10;
  m_N = 1;
  m_Epsilon = 0.0001;
  m_Sigma = 1.5;
  m_Lambda = 1.0;
  m_Alpha = 1.0;
  m_MaxIterations = 100;
  m_SmoothPVMapSigma = 2.0;
  m_OutputAbsoluteLocation = false;
  m_TrackJacobian = true;  
  m_WriteMagnitudeOfDisplacementImage = false;
  m_WriteTSurfImage = false;
  m_WriteGradientImage = false;
  m_WriteVelocityImage = false;
  m_WriteTransformedMovingImage = false;
  m_SmoothPVMaps = false;
  m_UseGradientMovingImage = false;
  
  niftkitkDebugMacro(<<"RegistrationBasedCTEFilter():Constructed with m_M=" << m_M \
      << ", m_N=" << m_N \
      << ", m_Epsilon=" << m_Epsilon \
      << ", m_Sigma=" << m_Sigma \
      << ", m_Lambda=" << m_Lambda \
      << ", m_Alpha=" << m_Alpha \
      << ", m_MaxIterations=" << m_MaxIterations \
      << ", m_OutputAbsoluteLocation=" << m_OutputAbsoluteLocation \
      << ", m_TrackJacobian=" << m_TrackJacobian \
      << ", m_WriteMagnitudeOfDisplacementImage=" << m_WriteMagnitudeOfDisplacementImage \
      << ", m_WriteTSurfImage=" << m_WriteTSurfImage \
      << ", m_WriteGradientImage=" << m_WriteGradientImage \
      << ", m_WriteVelocityImage=" << m_WriteVelocityImage \
      << ", m_WriteTransformedMovingImage=" << m_WriteTransformedMovingImage \
      << ", m_SmoothPVMaps=" << m_SmoothPVMaps \
      << ", m_SmoothPVMapSigma=" << m_SmoothPVMapSigma \
      << ", m_UseGradientMovingImage=" << m_UseGradientMovingImage \
      );
}

template< class TInputImage, typename TScalarType > 
void
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "M = " << m_M << std::endl;
  os << indent << "N = " << m_N << std::endl;
  os << indent << "Epsilon = " << m_Epsilon << std::endl;
  os << indent << "Sigma = " << m_Sigma << std::endl;
  os << indent << "Lambda = " << m_Lambda << std::endl;
  os << indent << "Alpha = " << m_Alpha << std::endl;
  os << indent << "MaxIterations = " << m_MaxIterations << std::endl;
  os << indent << "OutputAbsoluteLocation = " << m_OutputAbsoluteLocation << std::endl;
  os << indent << "TrackJacobian = " << m_TrackJacobian << std::endl;  
  os << indent << "WriteMagnitudeOfDisplacementImage = " << m_WriteMagnitudeOfDisplacementImage << std::endl;
  os << indent << "WriteTSurfImage = " << m_WriteTSurfImage << std::endl;
  os << indent << "WriteGradientImage = " << m_WriteGradientImage << std::endl;
  os << indent << "WriteVelocityImage = " << m_WriteVelocityImage << std::endl;
  os << indent << "WriteTransformedMovingImage = " << m_WriteTransformedMovingImage << std::endl;
  os << indent << "SmoothPVMaps = " << m_SmoothPVMaps << std::endl;
  os << indent << "SmoothPVMapSigma = " << m_SmoothPVMapSigma << std::endl;
  os << indent << "UseGradientMovingImage = " << m_UseGradientMovingImage << std::endl;
}

template <typename TInputImage, typename TScalarType >
void 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::InitializeScalarImage(ImageType *image,
                        RegionType& region,
                        SpacingType& spacing,
                        OriginType& origin,
                        DirectionType& direction)
{
  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetDirection(direction);
  image->SetOrigin(origin);
  image->Allocate(); 
  
  image->FillBuffer(0);    
}

template <typename TInputImage, typename TScalarType >
void 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::InitializeVectorImage(VectorImageType* image, 
                  const VectorImageRegionType& region, 
                  const VectorImageSpacingType& spacing, 
                  const VectorImageOriginType& origin, 
                  const VectorImageDirectionType& direction)
{
  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetDirection(direction);
  image->SetOrigin(origin);
  image->Allocate(); 
  
  VectorPixelType zero;
  zero.Fill(0);
  image->FillBuffer(zero);
}

template <typename TInputImage, typename TScalarType >
std::string
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::GetFileName(int i, std::string image, std::string ext)
{
  return image + "." + niftk::ConvertToString((int)i) + ext;
}

template <typename TInputImage, typename TScalarType >
void
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::WriteDisplacementField(std::string filename)
{

  // Creating this locally, so memory will be released when filter goes out of scope.
  typename VectorImageWriterType::Pointer vectorWriter = VectorImageWriterType::New();
  
  if (m_OutputAbsoluteLocation)
    {
	  niftkitkDebugMacro(<<"WriteDisplacementField():Writing absolute location to:" << filename);
      vectorWriter->SetInput(m_FinalPhiImage);
    }
  else
    {
	  niftkitkDebugMacro(<<"WriteDisplacementField():Writing displacement field to:" << filename);

      // Using a filter, because its multi-threaded.
      typename SubtractImageFilterType::Pointer subtractPhiZeroFromPhiFilter = SubtractImageFilterType::New();
      subtractPhiZeroFromPhiFilter->SetInput(0, m_FinalPhiImage );
      subtractPhiZeroFromPhiFilter->SetInput(1, m_PhiZeroImage );
      subtractPhiZeroFromPhiFilter->UpdateLargestPossibleRegion();
      vectorWriter->SetInput(subtractPhiZeroFromPhiFilter->GetOutput());
    }

  vectorWriter->SetFileName(filename);
  vectorWriter->Update();
}

template <typename TInputImage, typename TScalarType >   
void
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::RemoveFile(std::string filename)
{
 
  if (!(remove(filename.c_str()) == 0))
    {
	  niftkitkErrorMacro("GenerateData():Failed to remove:" +  filename);
    }
  else
    {
	  niftkitkDebugMacro(<<"GenerateData():Successfully removed:" +  filename);
    }
}

template <typename TInputImage, typename TScalarType >
double
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::EvaluateVelocityField(VectorImageType* velocityFieldAtTimeT, double dt)
{
  // Implements first part of equation 2 in paper.
  // The velocity field is discretized n times, so this method is called n times.
  // We seek a minimally deforming mapping, so the total number (summed over n images) should be as small as possible.
  ImageRegionConstIterator<VectorImageType> vIterator(velocityFieldAtTimeT, velocityFieldAtTimeT->GetLargestPossibleRegion());
  
  double cost = 0;
  double tmp = 0;
  VectorPixelType pixel;
  
  for(vIterator.GoToBegin(); !vIterator.IsAtEnd(); ++vIterator)
    {
      pixel = vIterator.Get();
      tmp = 0;
      for (unsigned int i = 0; i < Dimension; i++)
        {
          tmp += (pixel[i]*pixel[i]);
        }
      cost += tmp;
    }
  
  niftkitkDebugMacro(<<"EvaluateVelocityField():dt=" << dt << ", cost=" << cost);
  return cost * dt;    
}

template <typename TInputImage, typename TScalarType >
double
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::EvaluateRegistrationSimilarity(VectorImageType* phi, ImageType* target, ImageType* source)
{
  // Implements second part of equation 2 in paper.
  // We seek to minimise the difference between target and source.
  // Argument target should be the white + grep matter PV map.
  // Argument source should be the white matter pv map.
  // i.e. target isn't interpolated, but source is.
  // i.e. the mapping is from the Pwg image to the Pw image.
  
  double cost = 0;
  double diff = 0;
  
  VectorPixelType phiPixel;
  PixelType targetPixel;
  PixelType sourcePixel;
  
  Point<TScalarType, Dimension> phiPoint;
  ContinuousIndex<TScalarType, Dimension> continousIndex; 
  
  ImageRegionConstIterator<ImageType> targetIterator(target, target->GetLargestPossibleRegion());
  ImageRegionConstIterator<VectorImageType> phiIterator(phi, phi->GetLargestPossibleRegion());
  
  typename LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
  interpolator->SetInputImage(source);
  
  for (targetIterator.GoToBegin(), phiIterator.GoToBegin();
       !targetIterator.IsAtEnd(); 
       ++targetIterator, ++phiIterator)
    {
      phiPixel = phiIterator.Get();
      targetPixel = targetIterator.Get();
      
      for (unsigned int j = 0; j < Dimension; j++)
        {
          phiPoint[j] = phiPixel[j]; 
        }
      
      if (source->TransformPhysicalPointToContinuousIndex(phiPoint, continousIndex))
        {
          sourcePixel = interpolator->Evaluate(phiPoint);
          diff = targetPixel - sourcePixel;
          cost += (diff * diff);
        }
    }
  
  niftkitkDebugMacro(<<"EvaluateRegistrationSimilarity():cost=" << cost);
  return cost;
}

template <typename TInputImage, typename TScalarType >
double 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::EvaluateCostFunction(double velocityFieldEnergy, double imageSimilarity)
{
  double cost = (1.0 - this->m_Alpha) * velocityFieldEnergy + (this->m_Alpha * imageSimilarity);
  
  niftkitkDebugMacro(<<"EvaluateCostFunction():" << cost << " = (1.0 - " << m_Alpha << ")*" << velocityFieldEnergy << " + (" << this->m_Alpha << ")*" << imageSimilarity);
  return cost;    
}

template <typename TInputImage, typename TScalarType >
void 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::CopyVectorField(VectorImageType* a, VectorImageType* b)
{
  ImageRegionConstIterator<VectorImageType> aIterator(a, a->GetLargestPossibleRegion());
  ImageRegionIterator<VectorImageType> bIterator(b, b->GetLargestPossibleRegion());
  
  for (aIterator.GoToBegin(), bIterator.GoToBegin(); !aIterator.IsAtEnd(); ++aIterator, ++bIterator)
    {
      bIterator.Set(aIterator.Get());
    }
}

template <typename TInputImage, typename TScalarType >
double 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::CalculateMaxMagnitude(VectorImageType* vec)
{
  double squaredMagnitude = 0;
  double maxSquaredMagnitude = 0;
  VectorPixelType vectorPixel;
  
  ImageRegionConstIterator<VectorImageType> iterator(vec, vec->GetLargestPossibleRegion());
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    {
      vectorPixel = iterator.Get();
      squaredMagnitude = vectorPixel.GetSquaredNorm();
      if (squaredMagnitude > maxSquaredMagnitude)
        {
          maxSquaredMagnitude = squaredMagnitude;
        }
    }
  return vcl_sqrt(maxSquaredMagnitude);
    
}

template <typename TInputImage, typename TScalarType >
double 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::CalculateMaxDisplacement(VectorImageType* phi)
{
  typename SubtractImageFilterType::Pointer subtractPhiZeroFromPhiFilter = SubtractImageFilterType::New();
  subtractPhiZeroFromPhiFilter->SetInput(0, phi);
  subtractPhiZeroFromPhiFilter->SetInput(1, m_PhiZeroImage);
  subtractPhiZeroFromPhiFilter->Modified(); 
  subtractPhiZeroFromPhiFilter->UpdateLargestPossibleRegion();

  return CalculateMaxMagnitude(subtractPhiZeroFromPhiFilter->GetOutput());
}


template <typename TInputImage, typename TScalarType >
void 
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::CalculateMinAndMaxJacobian(VectorImageType* phi, double &min, double& max)
{
  typename SubtractImageFilterType::Pointer subtractPhiZeroFromPhiFilter = SubtractImageFilterType::New();
  subtractPhiZeroFromPhiFilter->SetInput(0, phi);
  subtractPhiZeroFromPhiFilter->SetInput(1, m_PhiZeroImage);
  subtractPhiZeroFromPhiFilter->Modified(); 
  subtractPhiZeroFromPhiFilter->UpdateLargestPossibleRegion();
  
  typename JacobianFilterType::Pointer jacobianFilter = JacobianFilterType::New();
  jacobianFilter->SetInput(subtractPhiZeroFromPhiFilter->GetOutput());
  jacobianFilter->SetUseImageSpacingOn();
  jacobianFilter->Modified(); 
  jacobianFilter->UpdateLargestPossibleRegion();
  
  typename MinMaxJacobianType::Pointer minMaxJacobianCalculator = MinMaxJacobianType::New();
  minMaxJacobianCalculator->SetImage(jacobianFilter->GetOutput());          
  minMaxJacobianCalculator->Compute();          

  min = minMaxJacobianCalculator->GetMinimum();
  max = minMaxJacobianCalculator->GetMaximum();
  
  return;
}

template <typename TInputImage, typename TScalarType >   
void
RegistrationBasedCTEFilter< TInputImage, TScalarType >
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");

  // Inputs are.....
  typename ImageType::Pointer whiteMatterPvMap = static_cast< ImageType * >(this->ProcessObject::GetInput(0));       
  typename ImageType::Pointer whitePlusGreyPvMap = static_cast< ImageType * >(this->ProcessObject::GetInput(1));       
  typename ImageType::Pointer thicknessPriorMap = static_cast< ImageType * >(this->ProcessObject::GetInput(2));

  // All this does is extend the array of pointers.
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

  // Optionally smooth input PV maps.
  if (m_SmoothPVMaps)
    {
	  niftkitkDebugMacro(<<"GenerateData():Smoothing WM PV map using sigma=" << m_SmoothPVMapSigma);
      m_SmoothWMPVMapFilter->SetInput(whiteMatterPvMap);
      m_SmoothWMPVMapFilter->SetUseImageSpacingOn();
      m_SmoothWMPVMapFilter->SetVariance(m_SmoothPVMapSigma*m_SmoothPVMapSigma);
      m_SmoothWMPVMapFilter->UpdateLargestPossibleRegion();
      whiteMatterPvMap = m_SmoothWMPVMapFilter->GetOutput();
      niftkitkDebugMacro(<<"GenerateData():Done");
      
      niftkitkDebugMacro(<<"GenerateData():Smoothing GM+WM PV map using sigma=" << m_SmoothPVMapSigma);
      m_SmoothGMWMPVMapFilter->SetInput(whitePlusGreyPvMap);
      m_SmoothGMWMPVMapFilter->SetUseImageSpacingOn();
      m_SmoothGMWMPVMapFilter->SetVariance(m_SmoothPVMapSigma*m_SmoothPVMapSigma);
      m_SmoothGMWMPVMapFilter->UpdateLargestPossibleRegion();
      whitePlusGreyPvMap = m_SmoothGMWMPVMapFilter->GetOutput();
      niftkitkDebugMacro(<<"GenerateData():Done");
    }
  
  // These are class variables, so we can use them in WriteDisplacementField
  // 
  // m_FinalPhiImage and m_PhiZeroImage hold the absolute position in millimetres, NOT an offset.
  // So, in m_PhiZeroImage, each voxel has a vector (of length=Dimension), where each component (x,y,z), 
  // is equal to the millimetre coordinate of that voxel, and it remains unchanged.
  // Then, m_FinalPhiImage has the corresponding mapped voxel location, as it evolves.
  
  m_PhiZeroImageUninitialized = VectorImageType::New();  // phi in paper, only at time zero.
  m_FinalPhiImage = VectorImageType::New();              // create the final phi image.

  // Create the time varying images
  RegionType    inputRegion    = whiteMatterPvMap->GetLargestPossibleRegion();
  SizeType      inputSize      = inputRegion.GetSize();
  IndexType     inputIndex     = inputRegion.GetIndex();
  DirectionType inputDirection = whiteMatterPvMap->GetDirection();
  OriginType    inputOrigin    = whiteMatterPvMap->GetOrigin();
  SpacingType   inputSpacing   = whiteMatterPvMap->GetSpacing();
  niftkitkDebugMacro(<<"GenerateData():Input images are of size=" << inputSize << ", spacing=" << inputSpacing << ", origin=" << inputOrigin << ", inputDirection=\n" << inputDirection);

  // We have m_N steps in time, so we need m_N images for v
  VectorImagePointer* vArray = new VectorImagePointer[m_N];

  for (unsigned int i = 0; i < m_N; i++)
    {
      vArray[i] = VectorImageType::New();
      InitializeVectorImage(vArray[i], inputRegion, inputSpacing, inputOrigin, inputDirection);
      
      niftkitkDebugMacro(<<"GenerateData():Initialised vArray[" << i << "], with size=" << vArray[i]->GetLargestPossibleRegion().GetSize()
          << ", spacing=" << vArray[i]->GetSpacing() 
          << ", origin=" << vArray[i]->GetOrigin() 
          << ", direction=\n" << vArray[i]->GetDirection());
      
    }

  // But we only need 1 image for u and 1 image for phi.
  VectorImagePointer* uArray = new VectorImagePointer[1];
  uArray[0] = VectorImageType::New();
  InitializeVectorImage(uArray[0], inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Initialised uArray[0]");

  VectorImagePointer* phiArray = new VectorImagePointer[1];
  phiArray[0] = VectorImageType::New();
  InitializeVectorImage(phiArray[0], inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Initialised phiArray[0]");
  
  niftkitkDebugMacro(<<"GenerateData():Creating phi(0) and phi(1) images");
  InitializeVectorImage(m_PhiZeroImageUninitialized.GetPointer(), inputRegion, inputSpacing, inputOrigin, inputDirection);
  InitializeVectorImage(m_FinalPhiImage.GetPointer(), inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Done");

  // "Optimization method": Step 1. Set delta t = 1/M. 
  double dt = (double)1/(double)m_M;
    
  // Step 1. Set phi(x,0) to x.
  niftkitkDebugMacro(<<"GenerateData():Initializing phiZero image");

  // Remember this filter may be running "in place", so don't bother looking at m_PhiZeroImageUninitialized after this point.
  m_InitializePhiZeroFilter->SetInput(m_PhiZeroImageUninitialized);
  m_InitializePhiZeroFilter->Update();
  m_PhiZeroImage = m_InitializePhiZeroFilter->GetOutput();
  
  niftkitkDebugMacro(<<"GenerateData():Done");
  
  // "Optimization method": Step 2. In step 5, its implicit that we come back here, so we need loop variables.
  double lastCost = std::numeric_limits<TScalarType>::max();
  double thisCost = 1;
  double costVelocityField = 0;
  double costImageSimilarity = 0;
  double epsilonRatio = 1;
  unsigned int currentIteration = 0;
  unsigned int iterationsPerField = this->m_M / this->m_N;

  typename VectorPhiPlusDeltaTTimesVFilterType::Pointer updatePhiFilter = VectorPhiPlusDeltaTTimesVFilterType::New();
  typename DasGradientFilterType::Pointer gradientFilter = DasGradientFilterType::New();
  typename DasTransformImageFilterType::Pointer transformedImageFilter = DasTransformImageFilterType::New();

  niftkitkDebugMacro(<<"GenerateData():thisCost=" << thisCost << ", lastCost=" << lastCost);
  
  while (currentIteration < m_MaxIterations && epsilonRatio >= m_Epsilon && thisCost < lastCost)
    {

      if (currentIteration > 0)
        {
          lastCost = thisCost;
        }

      // Then integrate phi over each of N velocity fields.
      for (unsigned int i = 0; i < this->m_N; i++)
        {
          // "Optimization method": Step (3) (a). Integrate phi
          // where thickness is less than thickness prior
          
    	  niftkitkDebugMacro(<<"GenerateData():Integrating phi using v(" << i << ") where m=" << this->m_M << ", n=" << this->m_N << ", so taking m/n=" << iterationsPerField << " steps");
          
          if (i == 0)
            {
        	  niftkitkDebugMacro(<<"GenerateData():Iteration " << currentIteration \
                << ", step " << i << ", setting updatePhiFilter->SetInput(0) to phiZero image");
              
              updatePhiFilter->SetInput(0, m_PhiZeroImage);  
            }
          else
            {
        	  niftkitkDebugMacro(<<"GenerateData():Iteration " << currentIteration \
                << ", step " << i << ", setting updatePhiFilter->SetInput(0), to phiArray[0] image");
              updatePhiFilter->SetInput(0, phiArray[0]);   
            }
          
          updatePhiFilter->SetInput(1, vArray[i]);
          updatePhiFilter->SetTimeZeroTransformation(m_PhiZeroImage);
          updatePhiFilter->SetThicknessPrior(thicknessPriorMap);
          updatePhiFilter->SetDeltaT(dt);
          updatePhiFilter->SetNumberOfSteps(iterationsPerField);
          updatePhiFilter->SetSubtractSteps(false);
          updatePhiFilter->Modified();
          updatePhiFilter->UpdateLargestPossibleRegion();

          // We have to copy the result into phiArray.
          // So, phiArray[0] represents the transformation AFTER m_M/m_N integration steps.
          
          CopyVectorField(updatePhiFilter->GetOutput(), phiArray[0]);

          niftkitkDebugMacro(<<"GenerateData():Iteration " << currentIteration \
            << ", phiArray[0] has max displacement " << this->CalculateMaxDisplacement(updatePhiFilter->GetOutput()));
          
          // Optionally, write out min and max jacobian of phi image, after processing each velocity image.
          
          if (m_TrackJacobian)
            {
              
              double min, max;

              CalculateMinAndMaxJacobian(updatePhiFilter->GetOutput(), min, max);

              niftkitkDebugMacro(<<"GenerateData():iteration " << currentIteration \
                  << ", phiArray[0] after vArray[" << i \
                  << "], min jacobian=" << min \
                  << ", max jacobian=" << max \
                  );
            }

        } // end for each of N velocity fields
      
      // Calculate gradient, just for total transformation.
      niftkitkDebugMacro(<<"GenerateData():Updating u() at iteration " << currentIteration );
      gradientFilter->SetInput(0, whiteMatterPvMap);
      gradientFilter->SetInput(1, whitePlusGreyPvMap);
      gradientFilter->SetInput(2, thicknessPriorMap);
      gradientFilter->SetInput(3, updatePhiFilter->GetThicknessImage());
      gradientFilter->SetTransformation(updatePhiFilter->GetOutput());
      gradientFilter->SetReverseGradient(true);
      gradientFilter->SetUseGradientTransformedMovingImage(!m_UseGradientMovingImage);
      gradientFilter->Modified();
      gradientFilter->UpdateLargestPossibleRegion();

      CopyVectorField(gradientFilter->GetOutput(), uArray[0]);
          
      niftkitkDebugMacro(<<"GenerateData():iteration " << currentIteration \
        << ", uArray[0] has max displacement " << this->CalculateMaxMagnitude(gradientFilter->GetOutput()));

      // Optionally, write out the gradient image for each uArray[i] image.
      
      if (m_WriteGradientImage)
        {
          // Using filters as they are multi-threaded.
          // Creating filters locally so the memory is released when the filter goes out of scope.
          
          typename VectorMagnitudeFilterType::Pointer vectorMagnitudeFilter = VectorMagnitudeFilterType::New();
          vectorMagnitudeFilter->SetInput(gradientFilter->GetOutput());
          vectorMagnitudeFilter->Modified();
          vectorMagnitudeFilter->UpdateLargestPossibleRegion();
          
          double energy = 0;
          ImageRegionConstIterator<ImageType> magnitudeIterator(vectorMagnitudeFilter->GetOutput(), vectorMagnitudeFilter->GetOutput()->GetLargestPossibleRegion());
          for (magnitudeIterator.GoToBegin(); !magnitudeIterator.IsAtEnd(); ++magnitudeIterator)
            {
              energy += magnitudeIterator.Get();
            }
          
          std::string gradientFile = "tmp.gradient.mha";
          niftkitkDebugMacro(<<"GenerateData():Writing image u(0) with energy " << energy << " at iteration " << currentIteration << " to file " << gradientFile);

          typename VectorImageWriterType::Pointer vectorWriter = VectorImageWriterType::New();
          vectorWriter->SetInput(gradientFilter->GetOutput());
          vectorWriter->SetFileName(gradientFile);
          vectorWriter->Update();
          
        }

      if (m_WriteMagnitudeOfDisplacementImage)
        {
          // Again, using filters as they are multi-threaded.
          // Again, creating filters locally, so the memory is released when the filter goes out of scope.
      
          std::string displacementFile = "tmp.mag.nii";
          niftkitkDebugMacro(<<"GenerateData():Calculating displacement magnitude and writing to:" << displacementFile);

          typename SubtractImageFilterType::Pointer subtractPhiZeroFromPhiFilter = SubtractImageFilterType::New();
          subtractPhiZeroFromPhiFilter->SetInput(0, updatePhiFilter->GetOutput());
          subtractPhiZeroFromPhiFilter->SetInput(1, m_PhiZeroImage);
      
          typename VectorMagnitudeFilterType::Pointer vectorMagnitudeFilter = VectorMagnitudeFilterType::New();
          vectorMagnitudeFilter->SetInput(subtractPhiZeroFromPhiFilter->GetOutput());
          vectorMagnitudeFilter->Modified();
      
          typename ScalarImageWriterType::Pointer scalarWriter = ScalarImageWriterType::New();
          scalarWriter->SetFileName(displacementFile);
          scalarWriter->SetInput(vectorMagnitudeFilter->GetOutput());
          scalarWriter->Update();
          niftkitkDebugMacro(<<"GenerateData():Done");
        }
      
      // Optionally write out a tranformed moving image.
      transformedImageFilter->SetInput(whiteMatterPvMap);
      transformedImageFilter->SetDefaultValue(0);
      transformedImageFilter->SetPhiTransformation(updatePhiFilter->GetOutput());
      transformedImageFilter->SetFileName("tmp.tmi.nii");
      transformedImageFilter->SetWriteTransformedMovingImage(m_WriteTransformedMovingImage);
      transformedImageFilter->Modified();
      transformedImageFilter->UpdateLargestPossibleRegion();
      
      // Copy to output two.
      ImageRegionConstIterator<ImageType> tmiIterator(transformedImageFilter->GetOutput(), transformedImageFilter->GetOutput()->GetLargestPossibleRegion());
      ImageRegionIterator<ImageType> outputTMIIterator(outputTransformedMovingImage, outputTransformedMovingImage->GetLargestPossibleRegion());
      for (tmiIterator.GoToBegin(), outputTMIIterator.GoToBegin();
           !tmiIterator.IsAtEnd();
           ++tmiIterator, ++outputTMIIterator)
        {
          outputTMIIterator.Set(tmiIterator.Get());
        }
      
      // Now we need to update vArrays.
      costVelocityField=0;

      for (unsigned int i = 0; i < this->m_N; i++)
        {

          VectorImageSpacingType sigma;
          for (unsigned int j = 0; j < Dimension; j++)
            {
              sigma[j] = this->m_Sigma; 
            }

          niftkitkDebugMacro(<<"GenerateData():iteration " << currentIteration \
            << ", i=" << i \
            << ", uArray[0] max = " << this->CalculateMaxMagnitude(uArray[0]) \
            << ", vArray[" << i << "] max = " << this->CalculateMaxMagnitude(vArray[i]) \
            );

          niftkitkDebugMacro(<<"GenerateData():Calculating v(" << i << ")");

          typename ConvolveFilterType::Pointer smoothVelocityWithGaussianFilter = ConvolveFilterType::New();
          smoothVelocityWithGaussianFilter->SetInput(uArray[0]);
          smoothVelocityWithGaussianFilter->SetSigma(sigma);
          smoothVelocityWithGaussianFilter->Modified();
          smoothVelocityWithGaussianFilter->UpdateLargestPossibleRegion();
          
          typename VectorVPlusLambdaUFilterType::Pointer vectorVPlusLambdaUFilter = VectorVPlusLambdaUFilterType::New();
          vectorVPlusLambdaUFilter->SetInput(0, vArray[i]);
          vectorVPlusLambdaUFilter->SetInput(1, smoothVelocityWithGaussianFilter->GetOutput());
          vectorVPlusLambdaUFilter->SetLambda(this->m_Lambda);
          vectorVPlusLambdaUFilter->SetSubtractSteps(false);
          vectorVPlusLambdaUFilter->Modified();
          vectorVPlusLambdaUFilter->UpdateLargestPossibleRegion();
          
          // Optionally, write out the velocity image for each vArray[i] image.
          
          if (m_WriteVelocityImage)
            {

              typename VectorMagnitudeFilterType::Pointer vectorMagnitudeFilter = VectorMagnitudeFilterType::New();
              vectorMagnitudeFilter->SetInput(vectorVPlusLambdaUFilter->GetOutput());
              vectorMagnitudeFilter->Modified();
              vectorMagnitudeFilter->UpdateLargestPossibleRegion();
              
              double energy = 0;
              ImageRegionConstIterator<ImageType> magnitudeIterator(vectorMagnitudeFilter->GetOutput(), vectorMagnitudeFilter->GetOutput()->GetLargestPossibleRegion());
              for (magnitudeIterator.GoToBegin(); !magnitudeIterator.IsAtEnd(); ++magnitudeIterator)
                {
                  energy += magnitudeIterator.Get();
                }
              

              std::string velocityFile = "tmp.velocity." + niftk::ConvertToString((int)i) + ".mha";
              niftkitkDebugMacro(<<"GenerateData():Writing image v(" << i << ") with energy " << energy << " at iteration " << currentIteration << " to file " << velocityFile);
              typename VectorImageWriterType::Pointer vectorWriter = VectorImageWriterType::New();
              vectorWriter->SetFileName(velocityFile);
              vectorWriter->SetInput(vectorVPlusLambdaUFilter->GetOutput());
              vectorWriter->Update();
            }

           // Copy result back to v.
          CopyVectorField(vectorVPlusLambdaUFilter->GetOutput(), vArray[i]);
          
          // Calculate cost of this velocity field.
          costVelocityField += EvaluateVelocityField(smoothVelocityWithGaussianFilter->GetOutput(), dt);
          
        } // end for i 

      // Make sure that m_FinalPhiImage has final value for phi,
      // as we need it later in method WriteDisplacementField
      CopyVectorField(updatePhiFilter->GetOutput(), m_FinalPhiImage);
      
      // Calculate Jacobian
      std::string jacobianString = " ";
      
      if (m_TrackJacobian)
        {
          // Again, using filters as they are multi-threaded.
          // Again, creating filters locally, so the memory is released when the filter goes out of scope.
          
    	  niftkitkDebugMacro(<<"GenerateData():Calculating jacobian");
          
          double min, max;
          
          CalculateMinAndMaxJacobian(updatePhiFilter->GetOutput(), min, max);
          
          niftkitkDebugMacro(<<"GenerateData():Done");
          
          jacobianString = ", minJac=" + niftk::ConvertToString(min) \
            + ", maxJac=" + niftk::ConvertToString(max);
        }
                  
      costImageSimilarity = EvaluateRegistrationSimilarity(updatePhiFilter->GetOutput(), whitePlusGreyPvMap , whiteMatterPvMap );
      
      thisCost = EvaluateCostFunction(costVelocityField, costImageSimilarity);
      
      if (currentIteration != 0)
        {
          epsilonRatio = fabs((thisCost - lastCost) / thisCost);  
        }

      niftkitkInfoMacro(<<"GenerateData():[" << currentIteration \
          << "] cost=" << thisCost \
          << ", eps=" << epsilonRatio \
          << ", maxDisp=" << this->CalculateMaxDisplacement(updatePhiFilter->GetOutput()) \
          << jacobianString
          );

      currentIteration++;

    } // end while

  // Thickness Propogation.
  typename ImageType::Pointer gwiBinaryImage = static_cast< ImageType * >(this->ProcessObject::GetInput(3));

  // Step (1). We have Igwi, create Tsurf.
  typename ImageType::Pointer iTotal = ImageType::New();  // I_{total} in paper
  typename ImageType::Pointer iHit = ImageType::New();    // I_{hit} in paper
  typename ImageType::Pointer tSurf = ImageType::New();   // T_{surf} in paper
  typename VectorImageType::Pointer pushForward = VectorImageType::New();
  
  niftkitkDebugMacro(<<"GenerateData():Creating and initialising tSurf image");
  InitializeScalarImage(tSurf.GetPointer(), inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Done");

  niftkitkDebugMacro(<<"GenerateData():Creating and initialising iTotal image");
  InitializeScalarImage(iTotal.GetPointer(), inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Done");

  niftkitkDebugMacro(<<"GenerateData():Creating and initialising iHit image");
  InitializeScalarImage(iHit.GetPointer(), inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Done");

  niftkitkDebugMacro(<<"GenerateData():Creating and initialising pushForward image");
  InitializeVectorImage(pushForward.GetPointer(), inputRegion, inputSpacing, inputOrigin, inputDirection);
  niftkitkDebugMacro(<<"GenerateData():Done");
  
  // Step (1). Calculate thickness value for tSurf image.
  // This is the deformation norm image mentioned in paper.
  // We have to re-calculate the values as we are propogating the other way.
  
  niftkitkDebugMacro(<<"GenerateData():Calculating tSurf");
  CopyVectorField(m_PhiZeroImage, pushForward);
  
  for (int i = (int)(m_N-1); i >= 0; i--)
    {
      niftkitkDebugMacro(<<"GenerateData():Propogating tSurf in field v(" << i << ")");
      
      // Here, vArray[i] are all fixed, so we are just iterating 
      // through velocity fields backwards to get inverse.
      updatePhiFilter->SetInput(0, pushForward);
      updatePhiFilter->SetInput(1, vArray[i]);
      updatePhiFilter->SetTimeZeroTransformation(m_PhiZeroImage);
      updatePhiFilter->SetThicknessPrior(thicknessPriorMap);
      updatePhiFilter->SetDeltaT(dt);
      updatePhiFilter->SetNumberOfSteps(iterationsPerField);
      updatePhiFilter->SetSubtractSteps(true);
      updatePhiFilter->Modified();
      updatePhiFilter->UpdateLargestPossibleRegion();
      
      CopyVectorField(updatePhiFilter->GetOutput(), pushForward);
    }
  
  // Now, subtract final position from initial position to get thickness.
  // You could sum along the path as you integrate, but paper specifically mentions Euclidean distance, so we stick with that for now.
  typename SubtractImageFilterType::Pointer subtractPhiZeroFromPhiFilter = SubtractImageFilterType::New();
  subtractPhiZeroFromPhiFilter->SetInput(0, pushForward);
  subtractPhiZeroFromPhiFilter->SetInput(1, m_PhiZeroImage);
  subtractPhiZeroFromPhiFilter->Modified();
  subtractPhiZeroFromPhiFilter->UpdateLargestPossibleRegion();
  
  // Then calculate thickness and store in tsurf.
  ImageRegionConstIterator<ImageType> gwiIterator(gwiBinaryImage, gwiBinaryImage->GetLargestPossibleRegion());
  ImageRegionIterator<ImageType> tSurfIterator(tSurf, tSurf->GetLargestPossibleRegion());
  ImageRegionIterator<VectorImageType> differenceIterator(subtractPhiZeroFromPhiFilter->GetOutput(), subtractPhiZeroFromPhiFilter->GetOutput()->GetLargestPossibleRegion());

  double maxDistance = 0;
  double distance = 0;
  
  for (gwiIterator.GoToBegin(),
       tSurfIterator.GoToBegin(),
       differenceIterator.GoToBegin();
       !gwiIterator.IsAtEnd();
       ++gwiIterator,
       ++tSurfIterator,
       ++differenceIterator)
    {
      if (gwiIterator.Get() > 0)
        {
          distance = vcl_sqrt(differenceIterator.Get().GetSquaredNorm());
          tSurfIterator.Set(distance);
          
          if (distance > maxDistance)
            {
              maxDistance = distance;
            }
        }
      else
        {
          tSurfIterator.Set(0);
        }
    }
  
  niftkitkDebugMacro(<<"GenerateData():Done propogating tSurf, maxDistance=" << maxDistance);

  // optionally save tsurf.
  
  if (m_WriteTSurfImage)
    {
      std::string tsurfFileName = "tmp.tsurf.nii";
      typename ScalarImageWriterType::Pointer tsurfWriter = ScalarImageWriterType::New();
      tsurfWriter->SetInput(tSurf);
      tsurfWriter->SetFileName(tsurfFileName);
      tsurfWriter->Update();
    }

  // Step (2): Propogate
  Point<TScalarType, Dimension> phiPoint;
  ContinuousIndex<TScalarType, Dimension> continousIndex; 
  
  /** Interpolators. */
  typename LinearInterpolatorType::Pointer tSurfInterpolator = LinearInterpolatorType::New();
  tSurfInterpolator->SetInputImage(tSurf);
  
  typename LinearInterpolatorType::Pointer gwiInterpolator = LinearInterpolatorType::New();
  gwiInterpolator->SetInputImage(gwiBinaryImage);
  
  niftkitkDebugMacro(<<"GenerateData():Calculating iHit and iTotal");
  CopyVectorField(m_PhiZeroImage, pushForward);
  
  // We need to take small steps to get a smooth propogation.
  double stepSize = maxDistance / (double)this->m_M; // i.e. at the maximum thickness, the actual distance in millimetres for 10 steps
  int stepSizeFactor = (int)(20.0*stepSize + 1.0);   // round up.
  iterationsPerField *= stepSizeFactor;

  niftkitkDebugMacro(<<"GenerateData():stepSize=" << stepSize \
      << ", stepSizeFactor=" << stepSizeFactor \
      << ", iterationsPerField=" << iterationsPerField \
      );
  
  for (int i = 0; i < (int)(this->m_N); i++)
    {
      niftkitkDebugMacro(<<"GenerateData():Propogating iHit and iTotal, iteration:" << i);

      for (unsigned int j = 0; j < iterationsPerField; j++)
        {
          updatePhiFilter->SetInput(0, pushForward);
          updatePhiFilter->SetInput(1, vArray[i]);
          updatePhiFilter->SetTimeZeroTransformation(m_PhiZeroImage);
          updatePhiFilter->SetThicknessPrior(thicknessPriorMap);
          updatePhiFilter->SetDeltaT(dt/(double)stepSizeFactor);
          updatePhiFilter->SetNumberOfSteps(1);
          updatePhiFilter->SetSubtractSteps(false);
          updatePhiFilter->Modified();
          updatePhiFilter->UpdateLargestPossibleRegion();
          
          CopyVectorField(updatePhiFilter->GetOutput(), pushForward);
          
          IndexType indexInPhiImage;
          IndexType index;
          
          VectorPixelType phiPixel;
          
          ImageRegionIteratorWithIndex<VectorImageType> phiIterator(updatePhiFilter->GetOutput(), updatePhiFilter->GetOutput()->GetLargestPossibleRegion());
          
          for(phiIterator.GoToBegin();
              !phiIterator.IsAtEnd();
              ++phiIterator)
            {
              phiPixel = phiIterator.Get();
              indexInPhiImage = phiIterator.GetIndex();
              
              for (unsigned int k = 0; k < Dimension; k++)
                {
                  phiPoint[k] = phiPixel[k];
                }
              
              if (updatePhiFilter->GetOutput()->TransformPhysicalPointToContinuousIndex(phiPoint, continousIndex))
                {
                  for (unsigned int k = 0; k < Dimension; k++)
                    {
                      index[k] = (int)(continousIndex[k]+0.5);
                    }
                  iHit->SetPixel(indexInPhiImage, iHit->GetPixel(indexInPhiImage) + gwiBinaryImage->GetPixel(index));
                  iTotal->SetPixel(indexInPhiImage, iTotal->GetPixel(indexInPhiImage) + tSurf->GetPixel(index));
                }          
              
            } // end for each voxel, phiIterator          
          
        } // end for each step, j
      
    } // end for each timepoint, i
  
  niftkitkDebugMacro(<<"GenerateData():Done");
  
  // Step (3): Write output. i.e. tVol IS the output
  // We mask using the GM map, so that values don't go outside the GM map. 
  // THis is purely cosmetic, as the thickness values are calculated from the transformation
  // that best matches the WM to the GM+WM, we are only talking here about masking the propogation.
  niftkitkDebugMacro(<<"GenerateData():Writing tVol (the output)");

  ImageRegionIterator<ImageType> iHitIterator(iHit, iHit->GetLargestPossibleRegion());
  ImageRegionIterator<ImageType> iTotalIterator(iTotal, iTotal->GetLargestPossibleRegion());
  ImageRegionIterator<ImageType> outputIterator(outputThicknessImage, outputThicknessImage->GetLargestPossibleRegion());
  ImageRegionIterator<ImageType> gmwmIterator(whitePlusGreyPvMap, whitePlusGreyPvMap->GetLargestPossibleRegion());
  
  for (iHitIterator.GoToBegin(),
       iTotalIterator.GoToBegin(),
       gmwmIterator.GoToBegin(),
       outputIterator.GoToBegin();
       !iHitIterator.IsAtEnd();
       ++iHitIterator,
       ++iTotalIterator,
       ++gmwmIterator,
       ++outputIterator)
    {
      if (iHitIterator.Get() > 0 && gmwmIterator.Get() > 0)
        {
          outputIterator.Set((double)iTotalIterator.Get() / (double)iHitIterator.Get());
        }
      else
        {
          outputIterator.Set(0);  
        }
    }
  
  niftkitkDebugMacro(<<"GenerateData():Done");

  delete [] uArray;
  delete [] vArray;
  delete [] phiArray;
  
  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif
