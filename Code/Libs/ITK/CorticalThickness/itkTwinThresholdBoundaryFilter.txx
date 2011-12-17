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
#ifndef __itkTwinThresholdBoundaryFilter_txx
#define __itkTwinThresholdBoundaryFilter_txx

#include "itkTwinThresholdBoundaryFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

#include "itkLogHelper.h"

namespace itk
{

template <typename TImageType>
TwinThresholdBoundaryFilter<TImageType>
::TwinThresholdBoundaryFilter()
{
  m_ThresholdForInput1 = 0;
  m_ThresholdForInput2 = 0; 
  m_True = 1;
  m_False = 0;
  m_FullyConnected = true;
}

template <typename TImageType>
void 
TwinThresholdBoundaryFilter<TImageType>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "ThresholdForInput1:" << this->m_ThresholdForInput1 << std::endl;
  os << indent << "ThresholdForInput2:" << this->m_ThresholdForInput2 << std::endl;
  os << indent << "True:" << this->m_True << std::endl;
  os << indent << "False:" << this->m_False << std::endl;
  os << indent << "FullyConnected:" << this->m_FullyConnected << std::endl;
}

template <typename TImageType >  
void
TwinThresholdBoundaryFilter<TImageType>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");
  
  this->AllocateOutputs();
  
  typename ImageType::ConstPointer input1 = static_cast< ImageType * >(this->ProcessObject::GetInput(0));  
  typename ImageType::ConstPointer input2 = static_cast< ImageType * >(this->ProcessObject::GetInput(1));
  typename ImageType::Pointer outputImage = static_cast< ImageType * >(this->ProcessObject::GetOutput(0));
  
  ImageRegionConstIteratorWithIndex<ImageType> input1Iterator(input1, input1->GetLargestPossibleRegion());
  ImageRegionIterator<ImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  
  typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  typename NeighborhoodIteratorType::ConstIterator neighbourhoodIterator;
  radius.Fill(1);
  
  NeighborhoodIteratorType input2Iterator(radius, input2, input2->GetLargestPossibleRegion()); 
   
  input1Iterator.GoToBegin();
  input2Iterator.GoToBegin();
  outputIterator.GoToBegin();
  
  typename ImageType::SizeType  input1Size = input1->GetLargestPossibleRegion().GetSize();
  typename ImageType::IndexType input1Index;
  typename ImageType::IndexType input2Index;
  
  while (!input1Iterator.IsAtEnd() && !input2Iterator.IsAtEnd() && !outputIterator.IsAtEnd())
    {
      
      bool isOnBoundary = false;
      
      if (input1Iterator.Get() >= this->m_ThresholdForInput1)
        {

          input1Index = input1Iterator.GetIndex();
          
          bool isOnEdge = false;
          
          for (unsigned int i = 0; i < Dimension; i++)
            {
              if ((int)(input1Index[i]) == 0 || (int)(input1Index[i]) == (int)(input1Size[i] - 1))
                {
                  isOnEdge = true;
                  break;
                }
            }
          
          if (!isOnEdge)
            {
              if (this->m_FullyConnected)
                {
                  for (neighbourhoodIterator = input2Iterator.Begin();
                       neighbourhoodIterator != input2Iterator.End(); 
                       ++neighbourhoodIterator)
                    {
                      if (*(*neighbourhoodIterator) >= this->m_ThresholdForInput2 && *neighbourhoodIterator != input2Iterator.GetCenterPointer())
                        {
                          isOnBoundary = true;  
                        }
                    }
                }
              else
                {
                  for (unsigned int i = 0; i < Dimension; i++)
                    {
                      for (int j = -1; j <= 1; j+=2)
                        {
                          input2Index    = input1Index;
                          input2Index[i] = input1Index[i] + j;
                          
                          if (input2->GetPixel(input2Index) >= this->m_ThresholdForInput2)
                            {
                              isOnBoundary = true;
                            }
                        } // end for +/- 1                      
                    } // end for each Dimension                  
                } // end else (not fully connected)
            } // end if not on edge
        } // end if on image 1 boundary
      
      if (isOnBoundary) 
        {
          outputIterator.Set(this->m_True);  
        }
      else
        {
          outputIterator.Set(this->m_False);
        }
      
      ++input1Iterator;
      ++input2Iterator;
      ++outputIterator;
    }
  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif
