/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 05:46:51 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7444 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkImageUpdatePixelWiseSingleValueProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::ImageUpdatePixelWiseSingleValueProcessor()
: m_Value(0)
{
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_Value=" << m_Value << std::endl;
  os << indent << "m_List, size=" << m_List.size() << std::endl; 
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::ClearList()
{
  m_List.clear();
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::AddToList(IndexType &voxelIndex)
{
  m_List.push_back(voxelIndex);
}

template<class TPixel, unsigned int VImageDimension>
unsigned long int 
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::GetNumberOfVoxels()
{
  return m_List.size();
}

template<class TPixel, unsigned int VImageDimension>
std::vector<int> 
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::ComputeMinimalBoundingBox()
{
  IndexType minIndex;
  IndexType maxIndex;

  IndexType voxelIndex;
  for (int i = 0; i < 3; i++)
  {
    minIndex[i] = std::numeric_limits<int>::max();
    maxIndex[i] = std::numeric_limits<int>::min();
  }

  for (unsigned int i = 0; i < m_List.size(); i++)
  {
    voxelIndex = m_List[i];
    for (int j = 0; j < 3; j++)
    {
      if (voxelIndex[j] < minIndex[j])
      {
        minIndex[j] = voxelIndex[j];
      }
      if (voxelIndex[j] > maxIndex[j])
      {
        maxIndex[j] = voxelIndex[j];
      }
    }
  }

  std::vector<int> region;
  region.push_back(minIndex[0]);
  region.push_back(minIndex[1]);
  region.push_back(minIndex[2]);
  region.push_back(maxIndex[0] - minIndex[0] + 1);
  region.push_back(maxIndex[1] - minIndex[1] + 1);
  region.push_back(maxIndex[2] - minIndex[2] + 1);
  
  return region;
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension> 
::ApplyUpdateToAfterImage()
{
  RegionType regionOfInterest = this->GetDestinationRegionOfInterest();
  ImagePointer targetImage = this->GetAfterImage();
  
  if (targetImage.IsNull())
  {
    itkExceptionMacro(<< "Target image is NULL");
  }
  
  // The region of interest should match the target image
  // but in the general case, as long as it is smaller, we are ok.
  if (!targetImage->GetLargestPossibleRegion().IsInside(regionOfInterest))
  {
    itkExceptionMacro("Region of interest=\n" << regionOfInterest << ", is not inside target region=\n" << targetImage->GetLargestPossibleRegion() );
  }
  
  itkDebugMacro(<< "Updating region=\n" << regionOfInterest << ", using value=" << m_Value << ", and list of size=" << m_List.size());
  
  IndexType voxelIndex;
  
  for (unsigned long int i = 0; i < m_List.size(); i++)
  {
    voxelIndex = m_List[i];
    if (regionOfInterest.IsInside(voxelIndex))
    {
      targetImage->SetPixel(voxelIndex, m_Value);
    }
  }
    
  itkDebugMacro(<< "Updating done");
}

} // end namespace
