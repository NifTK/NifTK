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
#ifndef __itkJorgesInitializationRelaxStreamlinesFilter_txx
#define __itkJorgesInitializationRelaxStreamlinesFilter_txx

#include "itkJorgesInitializationRelaxStreamlinesFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk
{

template <class TImageType, typename TScalarType, unsigned int NDimensions> 
JorgesInitializationRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::JorgesInitializationRelaxStreamlinesFilter()
{
  this->m_InitializeBoundaries = true;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void 
JorgesInitializationRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
typename JorgesInitializationRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions >::OutputImagePixelType
JorgesInitializationRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
::GetInitilizedValue(
    InputScalarImagePixelType& segmentedValue,
    InputScalarImageType* segmentedImage, 
    InputScalarImageType* gmPVImage, 
    InputScalarImageIndexType& index, 
    bool& initializeLOImage, 
    bool& initializeL1Image, 
    bool& addToGreyList)
{
  initializeLOImage = false;
  initializeL1Image = false;
  addToGreyList = false;
  
  InputScalarImageSpacingType spacing = segmentedImage->GetSpacing();

  typename InputScalarImageType::RegionType region;
  typename InputScalarImageType::SizeType size;
  typename InputScalarImageType::IndexType startingIndex;
  typename InputScalarImageType::IndexType neighbourIndex;
    
  size.Fill(3);
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      startingIndex[i] = index[i] - 1;
    } 
  
  region.SetSize(size);
  region.SetIndex(startingIndex);
  
  ImageRegionConstIteratorWithIndex<InputScalarImageType> neighbourHoodSegmentedIterator(segmentedImage, region);
  ImageRegionConstIteratorWithIndex<InputScalarImageType> neighbourHoodPVIterator(gmPVImage, region);

  double tmp = 0;
  double distance = 0;
  double averageDistance = 0;
  int counter = 0;
  InputScalarImagePixelType neighbourSegmentedValue;
  InputScalarImagePixelType currentGMPV;
  InputScalarImagePixelType neighbourGMPV;
  
  double meanVoxelSize = 0;
  double minusHalfMeanVoxelSize = 0;
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      meanVoxelSize += spacing[i];
    }
  meanVoxelSize /= (double)NDimensions;
  minusHalfMeanVoxelSize = -0.5 * meanVoxelSize;
  
  currentGMPV = gmPVImage->GetPixel(index);
  
  for (neighbourHoodSegmentedIterator.GoToBegin(),
       neighbourHoodPVIterator.GoToBegin();
       !neighbourHoodSegmentedIterator.IsAtEnd(); 
       ++neighbourHoodSegmentedIterator,
       ++neighbourHoodPVIterator
       )
    {
      neighbourSegmentedValue = neighbourHoodSegmentedIterator.Get();
      neighbourIndex = neighbourHoodSegmentedIterator.GetIndex();

      if (neighbourSegmentedValue != segmentedValue)
        {
          if (segmentedValue == this->GetWhiteMatterLabel())
            {
              initializeLOImage = true;
            }
          else if (segmentedValue == this->GetExtraCerebralMatterLabel())
            {
              initializeL1Image = true;
            }

          neighbourGMPV = gmPVImage->GetPixel(neighbourIndex);  

          if (neighbourSegmentedValue != this->GetGreyMatterLabel())
            {
              // We must have CSF touching WM, or WM touching CSF.
              addToGreyList = true;
              
              // So we calculate a thickness based on voxel size, and GM PV.
              distance = 0;
              for (unsigned int i = 0; i < NDimensions; i++)
                {
                  tmp = (currentGMPV + neighbourGMPV)*spacing[i]*(fabs((double)(neighbourIndex[i] - index[i])));
                  distance += tmp*tmp;
                }
              distance = sqrt(distance);              
            }
          else 
            {
              // We must have WM touching GM, or CSF touching GM
              
              // Had to set this to 0.25, as very low numbers cause distance estimation to be noisy due to small denominator.
              if (currentGMPV < 0.25)
                {
                  distance = minusHalfMeanVoxelSize;
                }
              else if (currentGMPV < 0.5)
                {
                  // If the boundary is between grey matter voxel (the neighbour)
                  // and the current voxel (CSF/WM), we want a negative value between 0 and minusHalfMeanVoxelSize.
                  // So the term in brackets should be a fraction (between 0 and 1) multiplied by half mean voxel size.
                  distance = minusHalfMeanVoxelSize + (((0.5 - neighbourGMPV) * (meanVoxelSize) / (currentGMPV - neighbourGMPV)) * meanVoxelSize/2.0);
                }
              else
                {
                  distance = (((0.5 - currentGMPV) * (meanVoxelSize) / (0 - currentGMPV)) * meanVoxelSize/2.0);                  
                }
            }
          
          averageDistance += distance;
          counter++;
          
        } // end if
    } // end for
  
  // We are taking the mean average.
  if (counter > 0)
    {
      averageDistance /= (double)counter;
    }

  /*
  niftkitkDebugMacro(<< "GetInitilizedValue():\t sv=" << segmentedValue \
      << ", index=" << index \
      << ", ave=" << averageDistance \
      << ", iL0=" << initializeLOImage \
      << ", iL1=" << initializeLOImage \
      << ", aGL=" << addToGreyList \
      );
  */
  return averageDistance;
}

template <class TImageType, typename TScalarType, unsigned int NDimensions >
void
JorgesInitializationRelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>
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

  niftkitkDebugMacro(<<"InitializeBoundaries():greyMatter.size()=" << completeListOfGreyMatterPixels.size());
  
  // We are solving L0 image, and L1 image over the complete GM mask.
  L0greyList = completeListOfGreyMatterPixels;
  L1greyList = completeListOfGreyMatterPixels;

  bool initializeLOImage;
  bool initializeL1Image;
  bool addToGreyList;
  
  if(this->m_InitializeBoundaries)
    {
      InputScalarImageIndexType index;   
      InputScalarImagePixelType segmentedValue;

      OutputImagePixelType distance = 0;
      bool needsInitializing = false;
      
      InputScalarImagePointer segmentedImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(2));
      InputScalarImagePointer gmPVImage = static_cast< InputScalarImageType * >(this->ProcessObject::GetInput(3));

      typename InputScalarImageType::RegionType region;
      typename InputScalarImageType::SizeType size;
      typename InputScalarImageType::IndexType startingIndex;

      for (unsigned int i = 0; i < NDimensions; i++)
        {
          startingIndex[i] = 1;
          size[i] = segmentedImage->GetLargestPossibleRegion().GetSize()[i] - 2;
        }
      region.SetSize(size);
      region.SetIndex(startingIndex);
      
      ImageRegionConstIteratorWithIndex<InputScalarImageType> segmentedImageIterator(segmentedImage, region);
      ImageRegionConstIteratorWithIndex<InputScalarImageType> gmpvImageIterator(gmPVImage, region);

      for (segmentedImageIterator.GoToBegin(),
           gmpvImageIterator.GoToBegin();
           !segmentedImageIterator.IsAtEnd();
           ++segmentedImageIterator,
           ++gmpvImageIterator)
        {
          
          index = segmentedImageIterator.GetIndex();
          segmentedValue = segmentedImageIterator.Get();
          
          if (segmentedValue == this->GetExtraCerebralMatterLabel() || segmentedValue == this->GetWhiteMatterLabel())
            {
              needsInitializing = false;

              if (
                     (segmentedValue == this->GetExtraCerebralMatterLabel() && this->IsOnWMBoundary(segmentedImage, index, true))
                  || (segmentedValue == this->GetExtraCerebralMatterLabel() && this->IsOnGMBoundary(segmentedImage, index, true))
                  || (segmentedValue == this->GetWhiteMatterLabel() && this->IsOnCSFBoundary(segmentedImage, index, true))
                  || (segmentedValue == this->GetWhiteMatterLabel() && this->IsOnWMBoundary(segmentedImage, index, true))
                  )
                {
                  needsInitializing = true;
                }
              
              if (needsInitializing)
                {
                  
                  distance = this->GetInitilizedValue(
                      segmentedValue, 
                      segmentedImage, 
                      gmPVImage, 
                      index, 
                      initializeLOImage, 
                      initializeL1Image, 
                      addToGreyList);
                  
                  if (initializeLOImage)
                    {
                      L0Image->SetPixel(index, distance);       
                    }
                  if (initializeL1Image)
                    {
                      L1Image->SetPixel(index, distance);
                    }
                  if (addToGreyList)
                    {
                      completeListOfGreyMatterPixels.push_back(index);
                    }
                  
                } // end if (needsInitializing)
            } // end if CSF or WM
        } // end for each voxel
    } // end if(this->m_InitializeBoundaries)

  niftkitkDebugMacro(<<"InitializeBoundaries():greyMatter.size()=" << completeListOfGreyMatterPixels.size() \
      << ", L0greyList.size()=" << L0greyList.size() \
      << ", L1greyList.size()=" << L1greyList.size() \
      );
  
}

} // end namespace

#endif // __itkJorgesInitializationRelaxStreamlinesFilter_txx
