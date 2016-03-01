/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBaseCTESegmentationImageFilter_txx
#define __itkBaseCTESegmentationImageFilter_txx

#include "itkBaseCTESegmentationFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <niftkConversionUtils.h>

namespace itk
{
template <typename TImageType >  
BaseCTESegmentationFilter<TImageType>
::BaseCTESegmentationFilter()
{
  m_TrustPVMaps = false;
  niftkitkDebugMacro(<<"BaseCTESegmentationFilter():Constructed with" \
      << ", TrustPVMaps=" << m_TrustPVMaps);
}

template <typename TImageType >  
void 
BaseCTESegmentationFilter<TImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "TrustPVMaps:" << m_TrustPVMaps << std::endl;
}

template <typename TImageType >
void
BaseCTESegmentationFilter<TImageType>
::CheckOrAssignLabels()
{
  niftkitkDebugMacro(<<"CheckOrAssignLabels():Started");
  
  typename InputImageType::Pointer inputImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
    
  InputPixelType      tmp;
  InputPixelType      labelValues[3];
  unsigned long int   labelCounts[3];
  int                 foundLabels = 0;
  int                 i;
  bool                alreadyGotIt = false;
  
  labelCounts[0] = labelCounts[1] = labelCounts[2] = 0;

  ImageRegionConstIteratorWithIndex<InputImageType> iter(inputImage, inputImage->GetLargestPossibleRegion());
  
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
    {
                                                     
      tmp = iter.Value();
      
      if (foundLabels == 0)
        {
          // first one.
          labelValues[0] = tmp;
          labelCounts[0]++;
          foundLabels++;
        }
      else
        {
          // Check, if we are at max number of labels.
          if (foundLabels == 3 && tmp != labelValues[0]
            && tmp != labelValues[1] && tmp != labelValues[2])
            {
        	  niftkitkDebugMacro(<<"GenerateData():So far, the labes we have are:");
              for (unsigned int i = 0; i < 3; i++)
                {
            	  niftkitkDebugMacro(<<"GenerateData():Label[" << i << "]=" << labelValues[i] << ", count=" << labelCounts[i]);
                }
              niftkitkDebugMacro(<<"GenerateData():And now we have value:" << tmp << ", at index=" << iter.GetIndex());
              
              itkExceptionMacro(<< "Input label image has > 3 labels");
            }
          else
            {
              alreadyGotIt = false;
              for (i = 0; i < foundLabels; i++)
                {
                  if (labelValues[i] == tmp)
                    {
                      alreadyGotIt = true;
                      labelCounts[i]++;
                      break;
                    }
                }
              if (alreadyGotIt == false)
                {
                  // new label.
                  labelValues[foundLabels] = tmp;
                  labelCounts[foundLabels]++;
                  foundLabels++;
                }
            }
        }
    }
  
  for (i = 0; i < 3; i++)
    {
	  niftkitkDebugMacro(<<"GenerateData():Label[" << i << "]=" << labelValues[i] << ", count=" << labelCounts[i]);
    }

  if (this->m_UserHasSetTheLabelThresholds)
    {
	  niftkitkDebugMacro(<<"GenerateData():User has set the labels, I'm checking that they match the data");
      for (i = 0; i < 3; i++)
        {
          if (labelValues[i] != this->m_GreyMatterLabel
           && labelValues[i] != this->m_ExtraCerebralMatterLabel
           && labelValues[i] != this->m_WhiteMatterLabel)
             {
               itkExceptionMacro(<< "Label[" << i << "] doesn't equal any of the specified values");
             }
        }
    }
  else
    {
	  niftkitkDebugMacro(<<"GenerateData():User did not set labels, so I'm hunting for defaults");
      // We have 3 labels, but which is which?
      int minIndex = -1;
      int maxIndex = -1;
      int middleIndex = -1;
      InputPixelType min = std::numeric_limits<InputPixelType>::max();
      InputPixelType max = std::numeric_limits<InputPixelType>::min();
      
      for (i = 0; i < 3; i++)
        {
          if ((int)labelCounts[i] < min)
            {
              min = labelCounts[i];
              minIndex = i;
            } 
          if ((int)labelCounts[i] > max)
            {
              max = labelCounts[i];
              maxIndex = i;
            }
        }
      
      for (i = 0; i < 3; i++)
        {
          if (i != minIndex && i != maxIndex)
            {
              middleIndex = i;
            }
        }
      
      this->m_ExtraCerebralMatterLabel = labelValues[maxIndex];
      this->m_GreyMatterLabel = labelValues[minIndex];
      this->m_WhiteMatterLabel = labelValues[middleIndex];
    }
    
  niftkitkDebugMacro(<<"CheckOrAssignLabels():Finished, Grey Label=" << this->m_GreyMatterLabel << ", white label=" << this->m_WhiteMatterLabel << ", extra cerebral label=" << this->m_ExtraCerebralMatterLabel );
}

template <typename TImageType >  
void
BaseCTESegmentationFilter<TImageType>
::CheckPVMap(std::string name, const InputImageType *image)
{
  if (!m_TrustPVMaps && image != 0)
    {
	  niftkitkDebugMacro(<<"CheckPVMap():Checking PV for 0 <= value <= 1");
      
      ImageRegionConstIterator<InputImageType> iterator(image, image->GetLargestPossibleRegion());
      InputPixelType value;
      unsigned long int i = 0;
      iterator.GoToBegin();
      while(!iterator.IsAtEnd())
        {
          value = iterator.Get();
          if (value < 0 || value > 1)
            {
              itkExceptionMacro(<< "In map:" << name << ", pixel at position:" << niftk::ConvertToString((int)i) << " has value:" << value << " which is not between 0 and 1.");   
            }
          i++;
          ++iterator;
        }
    }
  else
    {
	  niftkitkDebugMacro(<<"CheckPVMap():Skipped, as we are trusting it is correct.");
    }
}

} // end namespace

#endif 
