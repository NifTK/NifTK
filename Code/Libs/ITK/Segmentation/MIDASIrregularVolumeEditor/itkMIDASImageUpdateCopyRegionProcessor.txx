/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASImageUpdateCopyRegionProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASImageUpdateCopyRegionProcessor<TPixel, VImageDimension>
::MIDASImageUpdateCopyRegionProcessor()
: m_SourceImage(0)
{
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdateCopyRegionProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_SourceImage=" << std::endl;
  os << indent.GetNextIndent() << m_SourceImage << std::endl; 
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdateCopyRegionProcessor<TPixel, VImageDimension>
::SetSourceRegionOfInterest(std::vector<int> &array)
{
  assert (array.size() == 6);
  
  typename ImageType::RegionType region;
  typename ImageType::IndexType regionIndex;
  typename ImageType::SizeType regionSize;
  
  regionIndex[0] = array[0];
  regionIndex[1] = array[1];
  regionIndex[2] = array[2];
  regionSize[0] = array[3];
  regionSize[1] = array[4];
  regionSize[2] = array[5];
  
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);
  
  this->SetSourceRegionOfInterest(region);
}

template<class TPixel, unsigned int VImageDimension>
void
MIDASImageUpdateCopyRegionProcessor<TPixel, VImageDimension> 
::ApplyUpdateToAfterImage()
{
  RegionType sourceRegionOfInterest = this->GetSourceRegionOfInterest();
  RegionType destinationRegionOfInterest = this->GetDestinationRegionOfInterest();
  
  // The "AfterImage" is a sub-region of the destination image that is being edited.
  ImagePointer targetImage = this->GetAfterImage();
  
  if (m_SourceImage.IsNull())
  {
    itkExceptionMacro(<< "Source image is NULL");
  }
  
  if (targetImage.IsNull())
  {
    itkExceptionMacro(<< "Target image is NULL");
  }
  
  // The region of interest should match the target image
  // but in the general case, as long as it is smaller, we are ok.
  if (!targetImage->GetLargestPossibleRegion().IsInside(destinationRegionOfInterest))
  {
    itkExceptionMacro("Destination region of interest=\n" << destinationRegionOfInterest << ", is not inside target region=\n" << targetImage->GetLargestPossibleRegion() );
  }
  
  PasteImagePointerType pasteImageFilter = PasteImageFilterType::New();
  pasteImageFilter->InPlaceOn();
  pasteImageFilter->SetSourceImage(m_SourceImage);
  pasteImageFilter->SetSourceRegion(sourceRegionOfInterest);
  pasteImageFilter->SetDestinationImage(targetImage);
  pasteImageFilter->SetDestinationIndex(destinationRegionOfInterest.GetIndex()); 
  pasteImageFilter->Update();
  
  // The memory address changes, due to InPlaceOn.
  this->SetAfterImage(pasteImageFilter->GetOutput());
}

} // end namespace
