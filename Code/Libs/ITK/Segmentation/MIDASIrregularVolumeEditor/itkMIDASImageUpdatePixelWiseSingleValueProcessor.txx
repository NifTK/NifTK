/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASImageUpdatePixelWiseSingleValueProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::MIDASImageUpdatePixelWiseSingleValueProcessor()
: m_Value(0)
, m_UpdateCalculated(false)
{
  m_Indexes.clear();
  m_Before.clear();
  m_After.clear();
}


template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_Value=" << m_Value << std::endl;
  os << indent << "m_UpdateCalculated=" << m_UpdateCalculated << std::endl;
  os << indent << "m_Indexes, size=" << m_Indexes.size() << std::endl; 
  os << indent << "m_Before, size=" << m_Before.size() << std::endl; 
  os << indent << "m_After, size=" << m_After.size() << std::endl;
}


template<class TPixel, unsigned int VImageDimension>
void
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::ClearList()
{
  m_Indexes.clear();
  m_Before.clear();
  m_After.clear();
  m_UpdateCalculated = false;
}


template<class TPixel, unsigned int VImageDimension>
void
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::AddToList(IndexType &voxelIndex)
{
  m_Indexes.push_back(voxelIndex);
}


template<class TPixel, unsigned int VImageDimension>
unsigned long int 
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::GetNumberOfVoxels()
{
  return m_Indexes.size();
}


template<class TPixel, unsigned int VImageDimension>
std::vector<int> 
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
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

  for (unsigned int i = 0; i < m_Indexes.size(); i++)
  {
    voxelIndex = m_Indexes[i];
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
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension>
::ApplyListToDestinationImage(const DataListType& list)
{
  Superclass::ValidateInputs();

  if (list.size() != m_Indexes.size())
  {
    itkExceptionMacro(<< "Index list and data list are different sizes which is definitely a programming bug.");
  }
    
  
  ImagePointer destination = this->GetDestinationImage();
  for (unsigned long int i = 0; i < m_Indexes.size(); i++)
  {
    destination->SetPixel(m_Indexes[i], list[i]);
  }
}

template<class TPixel, unsigned int VImageDimension>
void
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension> 
::Undo()
{
  this->ApplyListToDestinationImage(m_Before);
}


template<class TPixel, unsigned int VImageDimension>
void
MIDASImageUpdatePixelWiseSingleValueProcessor<TPixel, VImageDimension> 
::Redo()
{
  IndexType voxelIndex;
  
  if (!m_UpdateCalculated)
  {

    Superclass::ValidateInputs();
    ImagePointer destination = this->GetDestinationImage();

    m_Before.clear();
    m_After.clear();
    
    // First take a copy of any existing data, so we can Undo.
    for (unsigned long int i = 0; i < m_Indexes.size(); i++)
    {
      voxelIndex = m_Indexes[i];
      m_Before.push_back(destination->GetPixel(voxelIndex));
      m_After.push_back(m_Value);
    }
    
    m_UpdateCalculated = true;
  }
  
  this->ApplyListToDestinationImage(m_After);
}

} // end namespace
