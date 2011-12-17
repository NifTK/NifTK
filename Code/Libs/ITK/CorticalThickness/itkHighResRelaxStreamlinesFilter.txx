/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkHighResRelaxStreamlinesFilter_txx
#define __itkHighResRelaxStreamlinesFilter_txx

#include "itkHighResRelaxStreamlinesFilter.h"
#include "itkFiniteDifferenceVoxel.h"
#include "itkFiniteDifferenceVoxel.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"

#include "itkLogHelper.h"

namespace itk
{

template <class TImageType, typename TScalarType, unsigned int NDimensions> 
HighResRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::HighResRelaxStreamlinesFilter()
{
  m_VoxelMultiplicationFactor = 2;
  m_LaplacianMap = NULL;
  m_L0L1 = NULL;
  m_VectorInterpolator = VectorNearestNeighborInterpolateImageFunction< InputVectorImageType, TScalarType >::New();
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
HighResRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
HighResRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::IntializeBoundaries(
    InputScalarImageType* gmpvImage,
    InputVectorImageType* vectorImage
    )
{
  PointType point;
  ContinuousIndexType continuousIndex;  
  FiniteDifferenceVoxelType *fdVox;
  MapIteratorType iterator;
  
  bool isOnCSFBoundary = false;
  bool isOnWMBoundary = false;
  double distanceToCSF = 0;
  double distanceToWM = 0;
  double maxStepSize = 0;
  double minStepSize = this->GetStepSizeThreshold();
  double direction = 0;  
  unsigned long int numberOnCSFBoundaryInitialized = 0;
  unsigned long int numberOnWMBoundaryInitialized = 0;
  unsigned long int indexOfCurrentVoxel = 0;
  
  m_L0L1 = m_LaplacianMap;

  // Precalculate default thickness
  maxStepSize = 2*this->GetMaxStepSize(gmpvImage);
  
  niftkitkDebugMacro(<<"IntializeBoundaries():minStepSize=" << minStepSize << ", maxStepSize=" << maxStepSize);
  
  for (iterator = m_L0L1->begin(); iterator != m_L0L1->end(); iterator++)
    {
      fdVox = ((*iterator).second);
  
      indexOfCurrentVoxel = fdVox->GetVoxelArrayIndex();    
      isOnCSFBoundary = fdVox->GetIsNextToCSF();
      isOnWMBoundary = fdVox->GetIsNextToWM();
      distanceToCSF = 0;
      distanceToWM = 0;
      
      if (gmpvImage != NULL)
        {
          if (isOnCSFBoundary)
            {
              
              direction = 1;
              
              point = fdVox->GetVoxelPointInMillimetres();
              vectorImage->TransformPhysicalPointToContinuousIndex(point, continuousIndex);
                            
              distanceToCSF = this->LagrangianInitialisation(continuousIndex,
                                                             direction,
                                                             maxStepSize,
                                                             minStepSize,
                                                             vectorImage,
                                                             gmpvImage
                                                             );
              numberOnCSFBoundaryInitialized++;
            }
          
          if (isOnWMBoundary)
            {
              direction = -1;
              
              point = fdVox->GetVoxelPointInMillimetres();
              vectorImage->TransformPhysicalPointToContinuousIndex(point, continuousIndex);

              distanceToWM = this->LagrangianInitialisation(continuousIndex,
                                                             direction,
                                                             maxStepSize,
                                                             minStepSize,
                                                             vectorImage,
                                                             gmpvImage
                                                             );
              numberOnWMBoundaryInitialized++;
            }
        }
      
      if (isOnCSFBoundary && isOnWMBoundary)
        {
          double total = distanceToCSF + distanceToWM;

          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(0, total/2.0);        
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(1, total/2.0);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(0, false);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(1, false);
        }
      else if (isOnCSFBoundary && !isOnWMBoundary)
        {
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(0, 0);        
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(1, distanceToCSF);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(0, true);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(1, false);          
        }
      else if (!isOnCSFBoundary && isOnWMBoundary)
        {
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(0, distanceToWM);        
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(1, 0);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(0, false);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(1, true);          
        }
      else
        {
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(0, 0);        
          (*m_L0L1)[indexOfCurrentVoxel]->SetValue(1, 0);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(0, true);
          (*m_L0L1)[indexOfCurrentVoxel]->SetNeedsSolving(1, true);                    
        }
      
    }

  niftkitkDebugMacro(<<"IntializeBoundaries():numberOnWMBoundaryInitialized=" << numberOnWMBoundaryInitialized \
      << ", numberOnCSFBoundaryInitialized=" << numberOnCSFBoundaryInitialized \
      );
  niftkitkDebugMacro(<<"IntializeBoundaries():Thickness maps have size=" << m_L0L1->size());
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
HighResRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::SolvePDE(int boundaryNumber,
    InputScalarImageSpacingType& virtualSpacing,
    InputScalarImageType* scalarImage,
    InputScalarImageType* gmpvImage,
    InputVectorImageType* vectorImage
    )
{
  typedef typename FiniteDifferenceVoxelType::ContinuousIndexType ContinuousIndexType;
  unsigned int dimensionIndex = 0;
  unsigned int dimensionIndexForAnisotropicScaleFactors = 0;  
  FiniteDifferenceVoxelType *vox;
  FiniteDifferenceVoxelType *voxPlus;
  FiniteDifferenceVoxelType *voxMinus;
  ContinuousIndexType index;
  ContinuousIndexType indexPlus;
  ContinuousIndexType indexMinus;
  OutputImagePixelType value;
  OutputImagePixelType divisor;
  OutputImagePixelType multiplier;
  OutputImagePixelType initialValue;
  OutputImagePixelType componentMagnitude;
  OutputImagePixelType currentPixelEnergy;
  OutputImagePixelType currentFieldEnergy;
  OutputImagePixelType previousFieldEnergy = 0;
  OutputImagePixelType pixelPlus;
  OutputImagePixelType pixelMinus;
  InputVectorImagePixelType vectorPixel;
  OutputImageSpacingType multipliers;
  PointType virtualVoxelPointInMillimetres;
  unsigned long int indexOfCurrentVoxel = 0;
  unsigned long int currentIteration = 0;
  MapIteratorType iterator;
  
  m_VectorInterpolator->SetInputImage(vectorImage);
  
  // Pre-calculate this.
  initialValue = 1;
  for (dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
    {
      initialValue *= virtualSpacing[dimensionIndex];
    }
  
  // Pre-calculate multipliers
  for (dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
    {
      multiplier = 1;
      for (dimensionIndexForAnisotropicScaleFactors = 0; dimensionIndexForAnisotropicScaleFactors < NDimensions; dimensionIndexForAnisotropicScaleFactors++)
        {
          if (dimensionIndexForAnisotropicScaleFactors != dimensionIndex)
            {
              multiplier *= (virtualSpacing[dimensionIndexForAnisotropicScaleFactors]);
            }
        }
      multipliers[dimensionIndex] = multiplier;
    }

  niftkitkDebugMacro(<<"SolvePDE():Calculated initialisers initialValue=" << initialValue << ", multipliers=" << multipliers);

  m_L0L1 = m_LaplacianMap;
  
  niftkitkDebugMacro(<<"SolvePDE():Using map from Laplacian class");

  currentIteration = 0;
  double epsilonRatio = 1;

  niftkitkDebugMacro(<<"GenerateData():Starting with currentIteration=" << currentIteration \
      << ", maxIterations=" << this->m_MaximumNumberOfIterations \
      << ", epsilonRatio=" << epsilonRatio \
      << ", convergenceThreshold=" << this->m_EpsilonConvergenceThreshold \
      );
  
  // Start of main loop
  while (currentIteration < this->m_MaximumNumberOfIterations && epsilonRatio >= this->m_EpsilonConvergenceThreshold)
    {
      currentFieldEnergy = 0;
      
      for (iterator = m_L0L1->begin(); 
           iterator != m_L0L1->end(); 
           iterator++)
        {
          value = initialValue;
          divisor = 0;
          
          currentPixelEnergy = 0;

          vox = (*iterator).second;
          
          /*
          std::cerr << "boundaryNumber=" << boundaryNumber \
            << ", vox=" << vox \
            << ", isBoundary=" << vox->GetBoundary() \
            << ", isNextToWM=" << vox->GetIsNextToWM() \
            << ", isNextToCSF=" << vox->GetIsNextToCSF() \
            << ", index=" << vox->GetVoxelIndex() \
            << std::endl;
          */
          
          if (!vox->GetBoundary() && vox->GetNeedsSolving(boundaryNumber))
            {

              vectorImage->TransformPhysicalPointToContinuousIndex(vox->GetVoxelPointInMillimetres(), index);
              vectorPixel = m_VectorInterpolator->EvaluateAtContinuousIndex(index);

              /*
              std::cerr << "index=" << index << ", in mm=" << vox->GetVoxelPointInMillimetres() << ", vectorPixel=" << vectorPixel << std::endl;
              */
              
              for (dimensionIndex = 0; dimensionIndex < NDimensions; dimensionIndex++)
                {
                  voxPlus = m_L0L1->find(vox->GetPlus(dimensionIndex))->second;
                  voxMinus = m_L0L1->find(vox->GetMinus(dimensionIndex))->second;

                  componentMagnitude = fabs(vectorPixel[dimensionIndex]);
                  
                  multiplier = (multipliers[dimensionIndex] * componentMagnitude);
                  
                  if (boundaryNumber == 0)
                    {
                      // Note, the indexPlus and indexMinus
                      // are MEANT to be different between L1 and L0.    
                      if (vectorPixel[dimensionIndex] >= 0)
                        {
                          pixelMinus = voxMinus->GetValue(boundaryNumber);
                          value += multiplier * pixelMinus; 
                          currentPixelEnergy += (pixelMinus * pixelMinus);
                        }
                      else
                        {
                          pixelPlus = voxPlus->GetValue(boundaryNumber);
                          value += multiplier * pixelPlus;
                          currentPixelEnergy += (pixelPlus * pixelPlus);
                        }                                            
                    }
                  else
                    {
                      // Note, the indexPlus and indexMinus
                      // are MEANT to be different between L1 and L0.    

                      if (vectorPixel[dimensionIndex] >= 0)
                        {
                          pixelPlus = voxPlus->GetValue(boundaryNumber);
                          value += multiplier * pixelPlus; 
                          currentPixelEnergy += (pixelPlus * pixelPlus); 
                        }
                      else
                        {
                          pixelMinus = voxMinus->GetValue(boundaryNumber);
                          value += multiplier * pixelMinus;
                          currentPixelEnergy += (pixelMinus * pixelMinus);
                        }                      
                      
                      
                    }
                  // And dont forget the divisor
                  divisor += multiplier;      
                  
                }

              if (divisor > 0)
                {
                  value /= divisor;
                }
              else
                {
                  value = 0;
                }          
              
              indexOfCurrentVoxel = vox->GetVoxelArrayIndex();
              (*m_L0L1)[indexOfCurrentVoxel]->SetValue(boundaryNumber, value);       
              currentPixelEnergy = sqrt(currentPixelEnergy);
              currentFieldEnergy += currentPixelEnergy;              
            }
        }

      if (currentIteration != 0)
        {
          epsilonRatio = fabs((previousFieldEnergy - currentFieldEnergy) / previousFieldEnergy);  
        }

      niftkitkInfoMacro(<<"GenerateData():[" << currentIteration << "] currentFieldEnergy=" << currentFieldEnergy << ", previousFieldEnergy=" << previousFieldEnergy << ", epsilonRatio=" << epsilonRatio << ", epsilonTolerance=" << this->m_EpsilonConvergenceThreshold);
      previousFieldEnergy = currentFieldEnergy;
                  
      currentIteration++;
             
    }    
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
HighResRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Started");
  
  if (m_LaplacianMap == NULL)
    {
      itkExceptionMacro(<< "You haven't supplied the HighRes Laplacian map.");
    }
  
  this->AllocateOutputs();
  
  InputScalarImagePointer laplacianImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(0));
  InputVectorImagePointer normalImage = static_cast< InputVectorImageType * >(this->ProcessObject::GetInput(1));
  InputScalarImagePointer segmentedImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(2));
  InputScalarImagePointer gmpvImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(3));
  OutputImagePointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  // Set up the new spacing dimension, and a virtual size. 
  // Note we never actually create the image's memory
  InputScalarImageSizeType inputSize = laplacianImage->GetLargestPossibleRegion().GetSize();
  InputScalarImageSpacingType inputSpacing = laplacianImage->GetSpacing();
  InputScalarImageOriginType inputOrigin = laplacianImage->GetOrigin();

  niftkitkDebugMacro(<<"GenerateData():Input image size=" << inputSize \
      << ", spacing=" << inputSpacing \
      << ", origin=" << inputOrigin \
      );

  InputScalarImageRegionType virtualRegion;
  InputScalarImageIndexType virtualIndex;
  InputScalarImageSizeType virtualSize;
  InputScalarImageSpacingType virtualSpacing;
  InputScalarImageOriginType virtualOrigin;
  
  for (unsigned int i =0; i < this->Dimension; i++)
    {
      virtualSpacing[i] = inputSpacing[i]/(float)m_VoxelMultiplicationFactor;
      virtualSize[i] = (int)(inputSize[i]*inputSpacing[i]/virtualSpacing[i]);
      virtualOrigin[i] = inputOrigin[i]; //+(((float)(inputSize[i]-1.0)/2.0)*inputSpacing[i])-(((float)(virtualSize[i]-1.0)/2.0)*virtualSpacing[i]);
    }
  virtualIndex.Fill(0);
  virtualRegion.SetSize(virtualSize);
  virtualRegion.SetIndex(virtualIndex);
  
  InputScalarImagePointer virtualImage = InputScalarImageType::New();
  virtualImage->SetRegions(virtualRegion);
  virtualImage->SetSpacing(virtualSpacing);
  virtualImage->SetOrigin(virtualOrigin);
  virtualImage->SetDirection(laplacianImage->GetDirection());
  
  niftkitkDebugMacro(<<"GenerateData():Virtual image size=" << virtualImage->GetLargestPossibleRegion().GetSize() \
      << ", spacing=" << virtualImage->GetSpacing() \
      << ", origin=" << virtualImage->GetOrigin() \
      );

  /**
   * Solve the PDE for inner and outer boundary.
   */
  if (gmpvImage.IsNotNull())
    {
	  niftkitkDebugMacro(<<"GenerateData():We are doing Lagrangian Initialization");
      this->IntializeBoundaries(gmpvImage.GetPointer(),  normalImage.GetPointer());      
    }
  else
    {
	  niftkitkDebugMacro(<<"GenerateData():We are not doing Lagrangian Initialization");
    }
  this->SolvePDE(0, virtualSpacing, laplacianImage.GetPointer(), gmpvImage.GetPointer(), normalImage.GetPointer());
  this->SolvePDE(1, virtualSpacing, laplacianImage.GetPointer(), gmpvImage.GetPointer(), normalImage.GetPointer());

  /** 
   * Now the big question. How to get the result out?
   */
  
  OutputImagePointer counterImage = OutputImageType::New();
  counterImage->SetRegions(segmentedImage->GetLargestPossibleRegion());
  counterImage->SetDirection(segmentedImage->GetDirection());
  counterImage->SetSpacing(segmentedImage->GetSpacing());
  counterImage->SetOrigin(segmentedImage->GetOrigin());
  counterImage->Allocate();
  counterImage->FillBuffer(0);

  OutputImagePointer accumulationImage = OutputImageType::New();
  accumulationImage->SetRegions(segmentedImage->GetLargestPossibleRegion());
  accumulationImage->SetDirection(segmentedImage->GetDirection());
  accumulationImage->SetSpacing(segmentedImage->GetSpacing());
  accumulationImage->SetOrigin(segmentedImage->GetOrigin());
  accumulationImage->Allocate();
  accumulationImage->FillBuffer(0);

  /*
   * Iterate through map, plotting points into the accumulation image, and incrementing counter image.
   * The output will be accumulationImage / counterImage. i.e. take an average.
   */

  PointType point;
  ContinuousIndexType continuousIndex;
  OutputImageIndexType outputIndex;
  InputScalarImageIndexType inputIndex;
  InputScalarImagePixelType inputValue;
  MapIteratorType L0L1Iterator;
  FiniteDifferenceVoxelType *L0L1Vox;
  
  for (L0L1Iterator = m_L0L1->begin(); 
       L0L1Iterator != m_L0L1->end(); 
       L0L1Iterator++)
    {
      L0L1Vox = (*L0L1Iterator).second;
      continuousIndex = L0L1Vox->GetVoxelIndex();
      virtualImage->TransformContinuousIndexToPhysicalPoint(continuousIndex, point);
      outputImage->TransformPhysicalPointToIndex( point, outputIndex );
      accumulationImage->SetPixel(outputIndex, accumulationImage->GetPixel(outputIndex) + (L0L1Vox->GetValue(0) + L0L1Vox->GetValue(1)));
      counterImage->SetPixel(outputIndex, counterImage->GetPixel(outputIndex) + 1);
    }

  ImageRegionConstIteratorWithIndex<InputScalarImageType> inputIterator(segmentedImage, segmentedImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> accumulationIterator(accumulationImage, accumulationImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> counterIterator(counterImage, counterImage->GetLargestPossibleRegion());
  for (outputIterator.GoToBegin(), inputIterator.GoToBegin(), accumulationIterator.GoToBegin(), counterIterator.GoToBegin();
       !outputIterator.IsAtEnd();
       ++outputIterator, ++inputIterator, ++accumulationIterator, ++counterIterator)
    {
      inputValue = inputIterator.Get();
      inputIndex = inputIterator.GetIndex();
      
      if(inputValue == this->GetWhiteMatterLabel() || inputValue == this->GetExtraCerebralMatterLabel())
        {
          outputIterator.Set(0);
        }
      else
        {
          if (counterIterator.Get() > 0)
            {
              outputIterator.Set(accumulationIterator.Get() / counterIterator.Get());
            }
          else
            {
              outputIterator.Set(0);
            }
        }      
    }

  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif 
