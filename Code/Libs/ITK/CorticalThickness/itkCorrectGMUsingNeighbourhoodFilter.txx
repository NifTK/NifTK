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
#ifndef __itkCorrectGMUsingNeighbourhoodFilter_txx
#define __itkCorrectGMUsingNeighbourhoodFilter_txx

#include "itkCorrectGMUsingNeighbourhoodFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "ConversionUtils.h"

#include "itkLogHelper.h"

namespace itk
{
template <typename TImageType >  
CorrectGMUsingNeighbourhoodFilter<TImageType>
::CorrectGMUsingNeighbourhoodFilter()
{
  m_UseFullNeighbourHood = true;
  niftkitkDebugMacro(<<"CorrectGMUsingNeighbourhoodFilter():Constructed");
}

template <typename TImageType >  
void 
CorrectGMUsingNeighbourhoodFilter<TImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "UseFullNeighbourHood=" << m_UseFullNeighbourHood << std::endl; 
}

template <typename TImageType >  
void
CorrectGMUsingNeighbourhoodFilter<TImageType>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");
  
  typename InputImageType::Pointer inputSegmentedImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));       
  typename OutputImageType::Pointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  this->CheckInputsAndOutputsSameSize();
  
  // Make sure that input segmented image only has 3 values.
  this->CheckOrAssignLabels();
    
  // Allocate memory for output image
  this->AllocateOutputs();

  // Now the main algorithm.
  m_ListOfGreyMatterVoxelsBeforeCorrection.clear();
  m_ListOfGreyMatterVoxelsAfterCorrection.clear();
  
  InputPixelType segmentedValue;
  InputIndexType segmentedIndex;
  
  ImageRegionConstIteratorWithIndex<InputImageType> segmentedIterator(inputSegmentedImage, inputSegmentedImage->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  
  // By default output image = input image.
  for (segmentedIterator.GoToBegin(), outputIterator.GoToBegin();
       !segmentedIterator.IsAtEnd();
       ++segmentedIterator, ++outputIterator)
    {
      segmentedValue = segmentedIterator.Get();

      if (segmentedValue == this->GetGreyMatterLabel())
        {
          m_ListOfGreyMatterVoxelsBeforeCorrection.push_back(segmentedIterator.GetIndex());
        }
      outputIterator.Set(segmentedValue);
    }
  
  OutputImageRegionType region;
  OutputImageSizeType size;
  OutputImageIndexType index;
  size.Fill(3);
  m_NumberReclassified = 0;
  
  // Now we check, and for each WM voxel that bounds GM/CSF, the CSF is reclassified as GM.
  if (m_UseFullNeighbourHood)
    {

      for (segmentedIterator.GoToBegin(), outputIterator.GoToBegin();
           !segmentedIterator.IsAtEnd();
           ++segmentedIterator, ++outputIterator)
        {
          segmentedValue = segmentedIterator.Get();
          segmentedIndex = segmentedIterator.GetIndex();
          
          if (segmentedValue == this->GetWhiteMatterLabel())
            {
              for (unsigned int i = 0; i < OutputImageType::ImageDimension; i++)
                {
                  index[i] = segmentedIndex[i] - 1;
                }
              
              region.SetSize(size);
              region.SetIndex(index);

              ImageRegionConstIteratorWithIndex<InputImageType> neighbourHoodIterator(inputSegmentedImage, region);
              for (neighbourHoodIterator.GoToBegin(); !neighbourHoodIterator.IsAtEnd(); ++neighbourHoodIterator)
                {
                  if (neighbourHoodIterator.Get() == this->GetExtraCerebralMatterLabel())
                    {
                      if (outputImage->GetPixel(neighbourHoodIterator.GetIndex()) != this->GetGreyMatterLabel())
                        {
                          outputImage->SetPixel(neighbourHoodIterator.GetIndex(), this->GetGreyMatterLabel());
                          m_NumberReclassified++;
                        }
                    }
                } // end for    
            } // end if
        } // end for
    }
  else
    {
      for (segmentedIterator.GoToBegin(), outputIterator.GoToBegin();
           !segmentedIterator.IsAtEnd();
           ++segmentedIterator, ++outputIterator)
        {

          segmentedValue = segmentedIterator.Get();
          segmentedIndex = segmentedIterator.GetIndex();
          if (segmentedValue == this->GetWhiteMatterLabel())
            {
              for (unsigned int i = 0; i < TImageType::ImageDimension; i++)
                {
                  for (int j = -1; j <= 1; j+= 2)
                    {
                      OutputImageIndexType offset = segmentedIndex;
                      offset[i] += j;
                      
                      if (inputSegmentedImage->GetPixel(offset) == this->GetExtraCerebralMatterLabel())
                        {
                          if (outputImage->GetPixel(offset) != this->GetGreyMatterLabel())
                            {
                              outputImage->SetPixel(offset, this->GetGreyMatterLabel());
                              m_NumberReclassified++;                              
                            }
                        } 
                    } // end for
                } // end for
            } // end if
        } // end for      
    }
  
  for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
    {
      if (outputIterator.Get() == this->GetGreyMatterLabel())
        {
          m_ListOfGreyMatterVoxelsAfterCorrection.push_back(outputIterator.GetIndex());
        }
    }
  
  niftkitkDebugMacro(<<"GenerateData():Finished, started with:" << m_ListOfGreyMatterVoxelsBeforeCorrection.size() \
      << ", and finished with:" << m_ListOfGreyMatterVoxelsAfterCorrection.size() \
      << ", reclassifying " << m_NumberReclassified \
      );
}

} // end namespace

#endif // __itkCorrectGMUsingNeighbourhoodFilter_txx
