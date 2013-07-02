/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkLagrangianInitializedRelaxStreamlinesFilter_txx
#define __itkLagrangianInitializedRelaxStreamlinesFilter_txx

#include "itkLagrangianInitializedRelaxStreamlinesFilter.h"
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorLinearInterpolateImageFunction.h>

namespace itk
{
template <class TImageType, typename TScalarType, unsigned int NDimensions> 
LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::LagrangianInitializedRelaxStreamlinesFilter()
{
  m_StepSizeThreshold = 0.001;
  m_GreyMatterPercentage = 0.5;
  m_MaximumSearchDistance = 10;
  m_DefaultMaximumSearchDistance = true;
  m_GreyMatterPVInterpolator = LinearInterpolateImageFunction< InputScalarImageType, TScalarType >::New();
  m_NormalsInterpolator = VectorLinearInterpolateImageFunction< InputVectorImageType, TScalarType >::New();
  
  niftkitkDebugMacro(<<"LagrangianInitializedRelaxStreamlinesFilter():Constructed, " \
    << "m_StepSizeThreshold=" << m_StepSizeThreshold \
    << ", m_GreyMatterPercentage=" << m_GreyMatterPercentage \
    << ", m_MaximumSearchDistance=" << m_MaximumSearchDistance \
    << ", m_DefaultMaximumSearchDistance=" << m_DefaultMaximumSearchDistance \
    );
  
  this->SetInitializeBoundaries(true);
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "StepSizeThreshold:" << m_StepSizeThreshold << std::endl;
  os << indent << "GreyMatterPercentage:" << m_GreyMatterPercentage << std::endl;
  os << indent << "MaximumSearchDistance:" << m_MaximumSearchDistance << std::endl;
  os << indent << "DefaultMaximumSearchDistance:" << m_DefaultMaximumSearchDistance << std::endl;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
typename LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>::OutputImagePixelType
LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::LagrangianInitialisation(
    ContinuousIndexType& index,
    double& direction,
    double& defaultStepSize,
    double& minStepSize, 
    InputVectorImageType* vectorImage,
    InputScalarImageType* greyMatterPVMap
    )
{
  OutputImagePixelType      furthestValue;
  OutputImagePixelType      previousValue;
  OutputImagePixelType      resultantValue;
  InputScalarImagePointType startingPoint;
  InputScalarImagePointType iteratingPoint;
  InputScalarImagePointType previousPoint;  // i.e. left bracket of threshold
  InputScalarImagePointType furthestPoint;  // i.e. right bracket of threshold
  InputVectorImagePixelType vectorDirection;
  bool                      failedToBracket;
  double                    vectorLength;
  int                       iterations;
  
  typedef ContinuousIndex<TScalarType, NDimensions> ContinuousIndexType;
  ContinuousIndexType continuousIndex;
  
  // Connect interpolators
  m_GreyMatterPVInterpolator->SetInputImage(greyMatterPVMap);
  m_NormalsInterpolator->SetInputImage(vectorImage);
  
  // Sample the vector direction, which we need to interpolate for non-integer voxel co-ordinates.
  vectorDirection = m_NormalsInterpolator->EvaluateAtContinuousIndex(index);
  vectorDirection *= direction;

  // Abandon if vector has zero length, which should never happen.
  // If it does it means the Laplacian bit probably failed.
  if (vectorDirection.GetNorm() < 0.0000001)
    {
	  niftkitkWarningMacro("LagrangianInitialisation():Index:" << index << ", has no vector normal, so return distance of zero");
      return 0;
    }

  // Set, the starting point.
  greyMatterPVMap->TransformContinuousIndexToPhysicalPoint( index, startingPoint );
  
  // We find furthest point to definitely bracket the point we are looking for.
  // If this fails, we return the maxVoxelSize, which is set to be 1/2 the voxel diagonal length.
  iterations = 0;
  previousValue = 2;
  furthestValue = 1;
  iteratingPoint = startingPoint;
  failedToBracket = false;
  
  while(furthestValue > m_GreyMatterPercentage && !failedToBracket)
    {
      iteratingPoint += vectorDirection;
      iterations++;
      
      if (m_GreyMatterPVInterpolator->IsInsideBuffer(iteratingPoint))
        {
          greyMatterPVMap->TransformPhysicalPointToContinuousIndex(iteratingPoint, continuousIndex );
          
          previousValue = furthestValue;
          furthestValue = this->m_GreyMatterPVInterpolator->EvaluateAtContinuousIndex( continuousIndex );
          
          if (furthestValue > previousValue)
            {
              failedToBracket = true;
              
              niftkitkWarningMacro("LagrangianInitialisation():pvalue is going uphill:index=" << index \
                  << ", i=" << iterations \
                  << ", d=" << direction \
                  << ", v=" << vectorDirection \
                  << ", sp=" << startingPoint \
                  << ", ip=" << iteratingPoint \
                  << ", pv=" << previousValue \
                  << ", fv=" << furthestValue \
                  << ", so capped at:" << defaultStepSize << " mm" \
                  );
            }
        }
      else
        {
          failedToBracket = true;  
        }     
    }

  if (!failedToBracket)
    {
      furthestPoint = iteratingPoint;
      previousPoint = (iteratingPoint - vectorDirection);
      
      // In practice, the step above should only take 1 or 2 iterations.
      // Now we do a dichotomy search between previousPoint and furthest point.
      // I googled for dichotomy search, and couldnt find much, so the implementation 
      // below is my best guess.
      
      vectorLength = 1;
      while(vectorLength > minStepSize)
        {
          iteratingPoint.SetToMidPoint(previousPoint, furthestPoint);

          vectorLength = iteratingPoint.EuclideanDistanceTo(previousPoint);
                
          greyMatterPVMap->TransformPhysicalPointToContinuousIndex(iteratingPoint, continuousIndex );
          resultantValue = this->m_GreyMatterPVInterpolator->EvaluateAtContinuousIndex( continuousIndex );
          
          if (resultantValue > m_GreyMatterPercentage)
            {
              previousPoint = iteratingPoint; 
            }
          else
            {
              furthestPoint = iteratingPoint;
            }
        }
          
      // And set the final value
      vectorLength = startingPoint.EuclideanDistanceTo(iteratingPoint);

      if (vectorLength > m_MaximumSearchDistance)
        {
    	  niftkitkWarningMacro("LagrangianInitialisation():Index:" << index \
              << ", startingPoint=" << startingPoint \
              << ", iteratingPoint=" << iteratingPoint \
              << ", vectorLength=" << vectorLength \
              << ", exceeds m_MaximumSearchDistance=" << m_MaximumSearchDistance \
              << ", so capped at:" << defaultStepSize << " mm");
          vectorLength = defaultStepSize;
        }

    }
  else
    {
      vectorLength = defaultStepSize;
    }  
  return vectorLength;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::UpdateGMPVMap(
    std::vector<InputScalarImageIndexType>& listOfGreyMatterPixels,
    InputScalarImageType* segmentedImage,
    InputVectorImageType* vectorImage,
    InputScalarImageType* greyPVMap, 
    InputScalarImageType* editedGreyPVMap)
{
  niftkitkDebugMacro(<<"UpdateGMPVMap():Copying PV map");
  
  ImageRegionConstIteratorWithIndex<InputScalarImageType> greyPVMapIterator(greyPVMap, greyPVMap->GetLargestPossibleRegion());
  ImageRegionIterator<InputScalarImageType> editedPVMapIterator(editedGreyPVMap, editedGreyPVMap->GetLargestPossibleRegion());

  greyPVMapIterator.GoToBegin();
  editedPVMapIterator.GoToBegin();

  while(!greyPVMapIterator.IsAtEnd() && !editedPVMapIterator.IsAtEnd())
    {
      editedPVMapIterator.Set(greyPVMapIterator.Get()); 
      ++greyPVMapIterator;
      ++editedPVMapIterator;
    }
  niftkitkDebugMacro(<<"UpdateGMPVMap():Modifying PV map");

  unsigned int              i, j, k, l;
  OutputImageIndexType      index;
  InputScalarImagePixelType segmentedImageValue;
  InputScalarImagePixelType pvImageValue;
  InputScalarImagePixelType modifiedPVImageValue;
  InputScalarImagePixelType csfValue;
  InputScalarImagePixelType greyValue;
  OutputImageIndexType      csfIndex;
  OutputImageIndexType      csfNeighbourhoodIndexPlus;
  OutputImageIndexType      csfNeighbourhoodIndexMinus;
  InputVectorImagePixelType plusVector;
  InputVectorImagePixelType minusVector;
  InputScalarImagePixelType plusScalar;
  InputScalarImagePixelType minusScalar;

  csfValue = this->GetExtraCerebralMatterLabel();
  greyValue = this->GetGreyMatterLabel();
  
  niftkitkDebugMacro(<<"UpdateGMPVMap():csfValue is=" << csfValue << ", greyValue is=" << greyValue);
  
  // Now do section 2.3.2 of Bourgeat et. al. ISBI 2008, where we modify GM PV value
  // according to whether the normals are opposing each other.
  
  for (i = 0; i < listOfGreyMatterPixels.size(); i++)
    {
      index = listOfGreyMatterPixels[i];
      for (j = 0; j < this->Dimension; j++)
        {
          for (k = -1; k <= 1; k+=2)
            {
              csfIndex = index;
              csfIndex[j] += k;
              segmentedImageValue = segmentedImage->GetPixel(csfIndex);
              
              if (segmentedImageValue == csfValue)
                {
                  for (l = 0; l < this->Dimension; l++)
                    {
                      
                      csfNeighbourhoodIndexPlus =  csfIndex;
                      csfNeighbourhoodIndexMinus =  csfIndex;

                      csfNeighbourhoodIndexPlus[l] += 1;
                      csfNeighbourhoodIndexMinus[l] -= 1;
                      
                      plusVector = vectorImage->GetPixel(csfNeighbourhoodIndexPlus);
                      minusVector = vectorImage->GetPixel(csfNeighbourhoodIndexMinus);
                      
                      plusScalar = segmentedImage->GetPixel(csfNeighbourhoodIndexPlus);
                      minusScalar = segmentedImage->GetPixel(csfNeighbourhoodIndexMinus);
                      
                      // Im checking that the scalar value is grey because
                      // we only have tangent vectors for grey matter voxels.
                      // So, we are definitely looking for a CSF voxel surrounded by two
                      // grey matter voxels, with normals in opposing directions.
                      
                      if (plusScalar == greyValue && minusScalar == greyValue)
                        {
                          if ((minusVector[l] < 0 && plusVector[l] > 0)
                              || (minusVector[l] > 0 && plusVector[l] < 0))
                            {
                              pvImageValue = greyPVMap->GetPixel(csfIndex);
                              modifiedPVImageValue = pvImageValue * (1.0 / (1.0 + ((fabs(minusVector[l]) + fabs(plusVector[l]))/2.0)));
                              editedGreyPVMap->SetPixel(csfIndex, modifiedPVImageValue);


                              niftkitkDebugMacro(<<"InitializeBoundaries():Modified location:" << csfIndex \
                                  << ", from:" << pvImageValue \
                                  << ", to:" << modifiedPVImageValue \
                                  << ", plusVector:" << plusVector \
                                  << ", minusVector:" << minusVector);
                                  
                            }
                        }
                    }
                }
            }
        }
    }    
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
double
LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GetMaxStepSize(InputScalarImageType* image)
{
  InputScalarImageSpacingType spacing = image->GetSpacing();
  double maxStepSize = 0;

  for (unsigned int i = 0; i < NDimensions; i++)
    {
      maxStepSize += (spacing[i]*spacing[i]);
    }
  maxStepSize = sqrt(maxStepSize);
  maxStepSize /= 2.0;
  
  niftkitkDebugMacro(<<"GetMaxStepSize():maxStepSize=" << maxStepSize);
  
  return maxStepSize;  
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
LagrangianInitializedRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::InitializeBoundaries(
    std::vector<InputScalarImageIndexType>& completeListOfGreyMatterPixels,
    InputScalarImageType* scalarImage,
    InputVectorImageType* vectorImage,
    OutputImageType* L0Image,
    OutputImageType* L1Image,
    std::vector<InputScalarImageIndexType>& L0greyList,
    std::vector<InputScalarImageIndexType>& L1greyList
    )
{
  niftkitkDebugMacro(<<"InitializeBoundaries():Started");

  unsigned int i = 0;
  double direction = 0;
  OutputImageIndexType greyIndex;
  OutputImageRegionType region;
  OutputImageSizeType size;
  size.Fill(3);
  
  InputScalarImagePointer segmentedImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(2));
  InputScalarImagePointer gmPVMapImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(3));

  // Create a copy of the input GM PV map, as we need to modify it.
  // It should be the same size as the input scalar image.

  typename InputScalarImageType::Pointer modifiedGreyMatterPVMap = InputScalarImageType::New();
  modifiedGreyMatterPVMap->SetRegions(scalarImage->GetLargestPossibleRegion());
  modifiedGreyMatterPVMap->SetOrigin(scalarImage->GetOrigin());
  modifiedGreyMatterPVMap->SetSpacing(scalarImage->GetSpacing());
  modifiedGreyMatterPVMap->SetDirection(scalarImage->GetDirection());
  modifiedGreyMatterPVMap->Allocate();

  // Modify the Grey Matter map.
  this->UpdateGMPVMap(completeListOfGreyMatterPixels,
      segmentedImage,
      vectorImage,
      gmPVMapImage,
      modifiedGreyMatterPVMap);
  
  // work out maximum and minimum step size to start searching with.  
  double minStepSize = m_StepSizeThreshold;
  double maxStepSize = GetMaxStepSize(scalarImage);
  
  if (m_DefaultMaximumSearchDistance)
    {
      m_MaximumSearchDistance = 2*maxStepSize;
    }

  // Loop through all grey matter voxels.
  ContinuousIndexType continousIndex;
  
  unsigned long int numberOnCSFBoundary = 0;
  unsigned long int numberOnWMBoundary = 0;
  bool isOnCSFBoundary = false;
  bool isOnWMBoundary = false;
  OutputImagePixelType distanceToCSF = 0;
  OutputImagePixelType distanceToWM = 0;
  
  for (i = 0; i < completeListOfGreyMatterPixels.size(); i++)
    {
      greyIndex = completeListOfGreyMatterPixels[i];
      
      for (unsigned int j = 0; j < NDimensions; j++)
        {
          continousIndex[j] = greyIndex[j];
        }

      distanceToCSF = 0;
      distanceToWM = 0;
      isOnCSFBoundary = this->IsOnCSFBoundary(segmentedImage, greyIndex, false);
      isOnWMBoundary = this->IsOnWMBoundary(segmentedImage, greyIndex, false);
      
      if (isOnCSFBoundary)
        {
          direction = 1;
          
          distanceToCSF = this->LagrangianInitialisation(
            continousIndex,
            direction,
            maxStepSize,
            minStepSize,
            vectorImage,
            modifiedGreyMatterPVMap.GetPointer());
          
          numberOnCSFBoundary++;
          
        }
      
      if (isOnWMBoundary)
        {
          direction = -1;
          
          distanceToWM = this->LagrangianInitialisation(
            continousIndex,
            direction,
            maxStepSize,
            minStepSize,
            vectorImage,
            modifiedGreyMatterPVMap.GetPointer());  

          numberOnWMBoundary++;
          
        }
      
      if (isOnCSFBoundary && isOnWMBoundary)
        {
          double total = distanceToWM + distanceToCSF;
          
          L0Image->SetPixel(greyIndex, total);     
          L1Image->SetPixel(greyIndex, total);   
        }
      else if (isOnCSFBoundary && !isOnWMBoundary)
        {
          L1Image->SetPixel(greyIndex, distanceToCSF);     
          L0greyList.push_back(greyIndex);
        }
      else if (!isOnCSFBoundary && isOnWMBoundary)
        {
          L0Image->SetPixel(greyIndex, distanceToWM);             
          L1greyList.push_back(greyIndex);
        }
      else
        {
          L0greyList.push_back(greyIndex);
          L1greyList.push_back(greyIndex); 
        }      
    }
  
  niftkitkDebugMacro(<<"InitializeBoundaries():Finished, from " << completeListOfGreyMatterPixels.size() \
      << ", grey voxels, I initialized " << numberOnCSFBoundary \
      << ", on CSF boundary, and " << numberOnWMBoundary \
      << ", on WM boundary");
}

} // end namespace

#endif 
