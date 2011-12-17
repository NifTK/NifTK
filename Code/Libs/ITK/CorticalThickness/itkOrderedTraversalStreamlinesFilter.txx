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
#ifndef __itkOrderedTraversalStreamlinesFilter_txx
#define __itkOrderedTraversalStreamlinesFilter_txx

#include "itkOrderedTraversalStreamlinesFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include <algorithm>
#include <vector>

#include "itkLogHelper.h"

namespace itk
{
template <class TImageType, typename TScalarType, unsigned int NDimensions> 
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::OrderedTraversalStreamlinesFilter()
{
  BOUNDARY = 0;
  UNVISITED = 1;
  VISITED = 2;
  SOLVED = 3;  
  niftkitkDebugMacro(<<"OrderedTraversalStreamlinesFilter():Constructed");
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
typename OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>::OutputImagePixelType
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GetLaplacian(const InputScalarImageIndexType& index,
               const int& offset,
               const typename InputScalarImageType::Pointer& scalarImage)
{
  OutputImagePixelType result = scalarImage->GetPixel(index);
  
  if (offset > 0)
    {
      result = 1.0 - result;
    }

  return result;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
typename OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>::OutputImagePixelType
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::Solve(const InputScalarImageIndexType& index, 
        const int& offset, 
        const typename InputVectorImageType::Pointer& vectorImage,
        const typename OutputImageType::Pointer& distanceImage)
{
  InputVectorImagePixelType vectorPixel;
  InputScalarImageIndexType indexPlus;
  InputScalarImageIndexType indexMinus; 
  OutputImagePixelType      result = 1;
  OutputImagePixelType      divisor = 0;
  OutputImagePixelType      multiplier;
  OutputImagePixelType      componentMagnitude = 0;
  unsigned int              dimensionIndexForAnisotropicScaleFactors = 0;
  
  OutputImageSpacingType spacing = distanceImage->GetSpacing();
  
  vectorPixel = vectorImage->GetPixel(index);

  for (unsigned int dimensionIndex = 0; 
       dimensionIndex < Dimension; 
       dimensionIndex++)
    {

      indexPlus = index;
      indexMinus = index;

      indexPlus[dimensionIndex] += offset;
      indexMinus[dimensionIndex] -= offset;  

      componentMagnitude = fabs(vectorPixel[dimensionIndex]);

      multiplier = 1;

      for (dimensionIndexForAnisotropicScaleFactors = 0; dimensionIndexForAnisotropicScaleFactors < Dimension; dimensionIndexForAnisotropicScaleFactors++)
        {
          if (dimensionIndexForAnisotropicScaleFactors != dimensionIndex)
            {
              multiplier *= (spacing[dimensionIndexForAnisotropicScaleFactors]);
            }
        }

      multiplier *= componentMagnitude;
      
      if (vectorPixel[dimensionIndex] >= 0)
        {
          result += multiplier * distanceImage->GetPixel(indexPlus);
        }
      else
        {
          result += multiplier * distanceImage->GetPixel(indexMinus);
        }                  
      divisor += multiplier;                      
    }
              
  if (divisor != 0)
    {
      result /= divisor;
    }
  else
    {
	  niftkitkErrorMacro("Divisor should not equal zero? programming bug?");
      result = 0;
    }  
  return result;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
bool
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::IsNextToBoundary(const InputScalarImageIndexType& index, 
                   const typename InputScalarImageType::Pointer& scalarImage,
                   const typename StatusImageType::Pointer& statusImage,
                   const InputScalarImagePixelType& threshold)
{
  // Matt: I'm convinced we should be doing only a 6 neighbourhood, not a 27,
  // so we get the smallest number of points, i.e. those closest to boundary.
  // Also, the finite difference stuff in Solve method is only a 6 neighbourhood.
  
  InputScalarImageIndexType indexOffset;
  
  for (unsigned int dim = 0; dim < NDimensions; dim++)
    {
      for (int offset = -1; offset <= 1; offset+=2)
        {
          indexOffset = index;
          indexOffset[dim] += offset;
          
          if (statusImage->GetPixel(indexOffset) == BOUNDARY 
              && fabs(scalarImage->GetPixel(indexOffset) - threshold) < 0.0001)
            {
              return true;
            }
        }
    }  
  return false;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::DoOrderedTraversal(const int& vectorDirectionMultiplier,
                     const InputScalarImagePixelType& threshold,
                     const typename InputScalarImageType::Pointer& scalarImage,
                     const typename InputVectorImageType::Pointer& vectorImage,
                     const typename OutputImageType::Pointer& distanceImage)
{

  niftkitkDebugMacro(<<"DoOrderedTraversal():vectorDirectionMultiplier:" << vectorDirectionMultiplier << ", threshold:" << threshold);
  
  typename StatusImageType::Pointer statusImage = StatusImageType::New();            
  statusImage->SetSpacing(scalarImage->GetSpacing());
  statusImage->SetOrigin(scalarImage->GetOrigin());
  statusImage->SetRegions(scalarImage->GetLargestPossibleRegion());
  statusImage->SetDirection(scalarImage->GetDirection());
  statusImage->Allocate();
  niftkitkDebugMacro(<<"GenerateData():Set tmp status image to size:" << statusImage->GetLargestPossibleRegion().GetSize());

  InputScalarImageSizeType size = scalarImage->GetLargestPossibleRegion().GetSize();
  
  // [STEP 1] Initially tag all points in R as UNVISITED.
  // 
  // We additionally:
  // A.) Keep a list of grey matter pixels, for speed.
  // B.) Mark the grey matter pixels as UNVISITED and everything else as BOUNDARY
  // C.) Set both distance images to initialize at zero.

  ImageRegionConstIteratorWithIndex<InputScalarImageType> 
    scalarIterator(scalarImage, scalarImage->GetLargestPossibleRegion());
    
  ImageRegionIterator<OutputImageType> 
    distanceImageIterator(distanceImage, distanceImage->GetLargestPossibleRegion());
    
  ImageRegionIterator<StatusImageType>
    statusImageIterator(statusImage, statusImage->GetLargestPossibleRegion());

  std::vector<InputScalarImageIndexType> listOfGreyMatterPixels;                                                    
  listOfGreyMatterPixels.clear();

  InputScalarImagePixelType scalarPixel;
  
  for (scalarIterator.GoToBegin(), 
       distanceImageIterator.GoToBegin(), 
       statusImageIterator.GoToBegin();
       !scalarIterator.IsAtEnd(); 
       ++statusImageIterator,
       ++distanceImageIterator,
       ++scalarIterator)
    {      
      scalarPixel = scalarIterator.Get();  
      
      if (scalarPixel > this->m_LowVoltage && scalarPixel < this->m_HighVoltage)
        {
          listOfGreyMatterPixels.push_back(scalarIterator.GetIndex()); 
          statusImageIterator.Set(UNVISITED);      
        }
      else
        {
          statusImageIterator.Set(BOUNDARY);
        }

      distanceImageIterator.Set(0);             
    }

  niftkitkDebugMacro(<<"DoOrderedTraversal():Found:" << listOfGreyMatterPixels.size() << ", grey matter pixels");


  MinMap                    visitedMap;
  InputScalarImageIndexType index;
  InputScalarImageIndexType indexOffset;
  OutputImagePixelType      distance = 0;  
  OutputImagePixelType      laplacianValue = 0;
  OutputImagePixelType      smallestLaplacian = 0;  
  unsigned long int         solvedPixels = 0;
  unsigned long int         pixelNumber = 0;
  unsigned long int         totalNumberOfPixels = listOfGreyMatterPixels.size();
  unsigned char             status;
  
  // [STEP 2] Solve for L_0 at points next to the boundary \partial_{0}R,
  // (where L_0 = 0) and re-tag these points as VISITED.
  
  for (pixelNumber = 0; pixelNumber < totalNumberOfPixels; pixelNumber++)
    {
      index = listOfGreyMatterPixels[pixelNumber];

      // Checking 6 neighbourhood, not 27.
      if (IsNextToBoundary(index, scalarImage, statusImage, threshold))
          {
            distance = Solve(index,
                             vectorDirectionMultiplier,
                             vectorImage,
                             distanceImage);

            distanceImage->SetPixel(index, distance);            
            statusImage->SetPixel(index, VISITED);
            
            // We maintain a multi-map of Laplacian value and index.
            // This is sorted by Laplacian value, smallest first.
            laplacianValue = this->GetLaplacian(index, vectorDirectionMultiplier, scalarImage);
            visitedMap.insert(Pair(laplacianValue, index));
          }
    }

  niftkitkDebugMacro(<<"DoOrderedTraversal():Found:" << visitedMap.size() << " pixels on 6 connected boundary:" << threshold);

  // [STEP 5]: Stop if all points in R have been tagged SOLVED, else go to STEP 3.
  while (solvedPixels < totalNumberOfPixels && visitedMap.size() > 0)
    {

      // [STEP 3] Find the grid point, within the current list of VISITED points, 
      // with the smallest value of Laplacian computed so far. 
      // Remove this point from the list and re-tag it as SOLVED.
      
      MinMapIterator iterator = visitedMap.begin();
      smallestLaplacian = (*iterator).first;
      index = (*iterator).second;      
      visitedMap.erase(iterator);
      statusImage->SetPixel(index, SOLVED);
      solvedPixels++;

      // [STEP 4] Update the values of L_0 using (8) for whichever neighbours
      // of this grid point are not yet tagged as SOLVED. If any of these
      // neighbours are currently tagged as UNVISITED, re-tag them VISITED and
      // add them to the current list of VISITED points.
      
      for (unsigned int dim = 0; dim < NDimensions; dim++)
        {
          for (int offset = -1; offset <= 1; offset += 2)
            {
              indexOffset = index;
              indexOffset[dim] += offset;
              
              status = statusImage->GetPixel(indexOffset);

              if (status != SOLVED && status != BOUNDARY)
                {
                  distance = Solve(indexOffset,
                                   vectorDirectionMultiplier,
                                   vectorImage,
                                   distanceImage);
                  
                  // We are updating the distance of a neighbouring pixel, so we must update the map.
                  distanceImage->SetPixel(indexOffset, distance); 
                  
                  if (status == UNVISITED)
                    {
                      // If the status of the neighbour is UNVISITED, it 
                      // isn't currently in the map, so we can just add it.
                      statusImage->SetPixel(indexOffset, VISITED);
                      laplacianValue = this->GetLaplacian(indexOffset, vectorDirectionMultiplier, scalarImage);
                      visitedMap.insert(Pair(laplacianValue, indexOffset));  
                    }   
                  else if (status == VISITED)
                    {
                      // should already be in map, so we need to update the map value,
                      // but we will have a range of values, so you need to find the right one.

                      laplacianValue = this->GetLaplacian(indexOffset, vectorDirectionMultiplier, scalarImage);
                      
                      MinMapIterator minMapSearchStart = visitedMap.lower_bound(laplacianValue - 0.01);
                      MinMapIterator minMapSearchFinish = visitedMap.upper_bound(laplacianValue + 0.01);
                      
                      MinMapIterator rangeOfValuesIterator;
                                            
                      for (rangeOfValuesIterator  = minMapSearchStart; 
                           rangeOfValuesIterator != minMapSearchFinish;
                           rangeOfValuesIterator++)
                        {
                          if ((*rangeOfValuesIterator).second == indexOffset)
                            {
                              visitedMap.erase(rangeOfValuesIterator);
                              
                              laplacianValue = this->GetLaplacian(indexOffset, vectorDirectionMultiplier, scalarImage);
                              visitedMap.insert(Pair(laplacianValue, indexOffset));
                              break;
                            }
                        }                      
                    }                                                                                                              
                }              
            }
        }
    } 
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
OrderedTraversalStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Started");
  
  this->AllocateOutputs();
  
  // Get pointers to input and output.
  typename InputScalarImageType::Pointer scalarImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(0));
  typename InputVectorImageType::Pointer vectorImage = static_cast< InputVectorImageType * >(this->ProcessObject::GetInput(1));
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  // Create temporary working space, I'm calling them "distance" images,
  // i.e. buffers to iteratively work out L0 and L1.
  
  typename OutputImageType::Pointer distanceImageL0 = OutputImageType::New();
  distanceImageL0->SetSpacing(scalarImage->GetSpacing());
  distanceImageL0->SetOrigin(scalarImage->GetOrigin());
  distanceImageL0->SetRegions(scalarImage->GetLargestPossibleRegion());
  distanceImageL0->SetDirection(scalarImage->GetDirection());
  distanceImageL0->Allocate();
  niftkitkDebugMacro(<<"GenerateData():Set distanceImageL0 to size:" << distanceImageL0->GetLargestPossibleRegion().GetSize());

  typename OutputImageType::Pointer distanceImageL1 = OutputImageType::New();
  distanceImageL1->SetSpacing(scalarImage->GetSpacing());
  distanceImageL1->SetOrigin(scalarImage->GetOrigin());
  distanceImageL1->SetRegions(scalarImage->GetLargestPossibleRegion());
  distanceImageL1->SetDirection(scalarImage->GetDirection());
  distanceImageL1->Allocate();
  niftkitkDebugMacro(<<"GenerateData():Set distanceImageL1 to size:" << distanceImageL1->GetLargestPossibleRegion().GetSize());
            
  // For L0.
  DoOrderedTraversal(-1, this->m_LowVoltage,  scalarImage, vectorImage, distanceImageL0);
  
  // For L1.
  DoOrderedTraversal( 1, this->m_HighVoltage, scalarImage, vectorImage, distanceImageL1);
      
  niftkitkDebugMacro(<<"GenerateData():Combining L0 and L1");
  
  ImageRegionIterator<OutputImageType> 
    outputIterator(outputImage, outputImage->GetLargestPossibleRegion());

  ImageRegionConstIterator<OutputImageType> 
    distanceImageL0Iterator(distanceImageL0, distanceImageL0->GetLargestPossibleRegion());

  ImageRegionConstIterator<OutputImageType> 
    distanceImageL1Iterator(distanceImageL1, distanceImageL1->GetLargestPossibleRegion());

  OutputImagePixelType thickness;
  
  for (outputIterator.GoToBegin(), 
       distanceImageL0Iterator.GoToBegin(), 
       distanceImageL1Iterator.GoToBegin(); 
       !outputIterator.IsAtEnd(); 
       ++outputIterator, 
       ++distanceImageL0Iterator, 
       ++distanceImageL1Iterator)
    {
      thickness = distanceImageL0Iterator.Get() + distanceImageL1Iterator.Get();
      outputIterator.Set(thickness);
    }

  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif // __itkImageRegistrationFilter_txx
