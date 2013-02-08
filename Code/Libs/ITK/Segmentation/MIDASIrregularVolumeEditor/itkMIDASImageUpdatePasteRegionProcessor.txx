/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASImageUpdatePasteRegionProcessor.h"
#include "itkImageRegionIterator.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASImageUpdatePasteRegionProcessor<TPixel, VImageDimension>
::MIDASImageUpdatePasteRegionProcessor()
: m_SourceImage(NULL)
, m_CopyBackground(true)
{
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdatePasteRegionProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);   
  os << indent << "m_SourceImage=" << std::endl;
  os << indent.GetNextIndent() << m_SourceImage << std::endl;
  os << indent << "m_CopyBackground=" << m_CopyBackground << std::endl;   
  os << indent << "m_SourceRegionOfInterest=" <<std::endl;
  os << indent.GetNextIndent() << m_SourceRegionOfInterest << std::endl;
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdatePasteRegionProcessor<TPixel, VImageDimension>
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
MIDASImageUpdatePasteRegionProcessor<TPixel, VImageDimension> 
::ApplyUpdateToAfterImage()
{
  RegionType sourceRegionOfInterest = this->GetSourceRegionOfInterest();
  RegionType destinationRegionOfInterest = this->GetDestinationRegionOfInterest();
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
  
  ImageRegionConstIterator<ImageType> sourceIterator(m_SourceImage, m_SourceRegionOfInterest);
  ImageRegionIterator<ImageType> targetIterator(targetImage, destinationRegionOfInterest);
  
  for (sourceIterator.GoToBegin(), targetIterator.GoToBegin(); !sourceIterator.IsAtEnd() && !targetIterator.IsAtEnd(); ++sourceIterator, ++targetIterator)
  {
    if (m_CopyBackground || sourceIterator.Get() > 0)
    {
      targetIterator.Set(sourceIterator.Get());
    }
  }   
}

} // end namespace
