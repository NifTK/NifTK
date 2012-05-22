/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASThresholdAndAxialCutoffFilter_txx
#define itkMIDASThresholdAndAxialCutoffFilter_txx

#include "itkMIDASThresholdAndAxialCutoffFilter.h"
#include "itkImageRegionIterator.h"

#include <map>
#include <algorithm>

namespace itk
{
  //Constructor
  template <class TImage>
  MIDASThresholdAndAxialCutoffFilter<TImage>::MIDASThresholdAndAxialCutoffFilter()
  {
    this->SetNumberOfRequiredOutputs(1);
    m_UseRegionToProcess = false;
  }


  template <class TImage>
  void MIDASThresholdAndAxialCutoffFilter<TImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_UseRegionToProcess=" << m_UseRegionToProcess << std::endl;
    os << indent << "m_LowerThreshold=" << m_LowerThreshold << std::endl;
    os << indent << "m_UpperThreshold=" << m_UpperThreshold << std::endl;
    os << indent << "m_InsideRegionValue=" << m_InsideRegionValue << std::endl;
    os << indent << "m_OutsideRegionValue=" << m_OutsideRegionValue << std::endl;
    os << indent << "m_RegionToProcess=" << m_RegionToProcess << std::endl;
  }


  template <class TImage>
  void MIDASThresholdAndAxialCutoffFilter<TImage>::BeforeThreadedGenerateData()
  {   
    const unsigned int numberOfInputImages = this->GetNumberOfInputs(); 
    
    if(numberOfInputImages != 1)
    {
      itkExceptionMacro(<< "There should be exactly one input image for MIDASThresholdAndAxialCutoffFilter. ");
    }
    
    // Check input is set.
    InputImageType *inputImage = static_cast<InputImageType*>(this->ProcessObject::GetInput(0));
    if(!inputImage)
    {
      itkExceptionMacro(<< "Input is not set!");
    }
    
    //Get the input image region
    InputImageRegionType inputImageRegion = inputImage->GetLargestPossibleRegion();
    
    //Check if the dimensions are correct
    if(m_RegionToProcess.GetImageDimension() != inputImageRegion.GetImageDimension())
    {
      itkExceptionMacro(<< "Image Dimensions does not match for the input image and the region to process.");
    }
    
    //check if the region is smaller than or equal in size to the input image
    for(unsigned long int i = 0; i < TImage::ImageDimension; i++)
    {
      if(m_RegionToProcess.GetSize(i) > inputImageRegion.GetSize(i))
      {
        itkExceptionMacro(<< "Region size is greater than the input image.");
      }
    }

    //check if the region is within the input image
    for(unsigned long int i = 0; i < TImage::ImageDimension; i++)
    {
      if( (m_RegionToProcess.GetIndex(i) + m_RegionToProcess.GetSize(i))  
          > (inputImageRegion.GetIndex(i) + inputImageRegion.GetSize(i)) ) 
      {
        itkExceptionMacro(<< "Region does not lie inside the input image.");
      }
    }

    // Get the output pointer
    OutputImagePointer outputImagePtr = this->GetOutput();

    typedef ImageRegionIterator<OutputImageType> OutputImageIterator;
    OutputImageIterator  outputImageIter(outputImagePtr, outputImagePtr->GetLargestPossibleRegion());
    
    //checking one pixel at a a time
    while(!outputImageIter.IsAtEnd())
    {
      outputImageIter.Set(m_OutsideRegionValue);
      ++outputImageIter;  
    }

  }


  template <class TImage>
  void MIDASThresholdAndAxialCutoffFilter<TImage>
    ::ThreadedGenerateData(const OutputImageRegionType &outputRegionForThread, int ThreadID)
  {   
    // Get the input and output pointers
    InputImageConstPointer  inputImagePtr  = this->GetInput();
    OutputImagePointer outputImagePtr = this->GetOutput();

    // Define an iterator that will walk the input image for this thread
    typedef ImageRegionConstIterator<InputImageType> InputImageIterator;
    typedef ImageRegionIterator<OutputImageType> OutputImageIterator;

    InputImageIterator  inputImageIter(inputImagePtr, outputRegionForThread);
    OutputImageIterator  outputImageIter(outputImagePtr, outputRegionForThread);
    
    IndexType regionIndex = m_RegionToProcess.GetIndex();
    SizeType regionSize   = m_RegionToProcess.GetSize();

    inputImageIter.GoToBegin();
    outputImageIter.GoToBegin();

    bool withinRegion = false;
    IndexType imageIndex;
    PixelType currentPixelValue;
    
    //checking one pixel at a a time
    while(!outputImageIter.IsAtEnd())
    {
      withinRegion = true;
      imageIndex = inputImageIter.GetIndex();

      if(m_UseRegionToProcess)
      {
        //check to see this is within the region to process
        for(unsigned long int i = 0; i < TImage::ImageDimension; i++)
        {
          //when on the same axis, compare the other axis coordinates
          //if the pixel is out of the region, then break out and go for the next pixel
          if( (imageIndex[i] < regionIndex[i]) ||
              ( (imageIndex[i] > regionIndex[i]) && (imageIndex[i] >= (regionIndex[i] + (int)regionSize[i])) ) )
          {
            //then the pixel index is outside the region
            withinRegion = false;
            break;
          }
        }//end of "for" loop for all dimensions of a pixel
      
        if(!withinRegion)
        {
          ++inputImageIter;
          ++outputImageIter;
          continue;
        }
      }
          
      //now check to see whether it falls within the threshold values
      currentPixelValue = inputImageIter.Get();
      if((currentPixelValue >= m_LowerThreshold) && (currentPixelValue <= m_UpperThreshold))
      {
        outputImageIter.Set(m_InsideRegionValue); 
      }
      else
      {
        outputImageIter.Set(m_OutsideRegionValue); 
      }

      ++inputImageIter;
      ++outputImageIter;
      
    }//end of while loop
  }    

}//end namespace itk


#endif
