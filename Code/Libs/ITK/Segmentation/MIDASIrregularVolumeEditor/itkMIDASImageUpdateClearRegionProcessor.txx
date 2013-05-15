/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASImageUpdateClearRegionProcessor.h"
#include <itkImageRegionIterator.h>

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASImageUpdateClearRegionProcessor<TPixel, VImageDimension>
::MIDASImageUpdateClearRegionProcessor()
: m_WipeValue(0)
{
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdateClearRegionProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_WipeValue=" << m_WipeValue << std::endl; 
}

template<class TPixel, unsigned int VImageDimension>
void
MIDASImageUpdateClearRegionProcessor<TPixel, VImageDimension> 
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
  
  itkDebugMacro(<< "Wiping region=\n" << regionOfInterest);
  
  ImageRegionIterator<ImageType> iterator(targetImage, regionOfInterest);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    iterator.Set(m_WipeValue);
  }
  
  itkDebugMacro(<< "Wiping done");
  
}

} // end namespace
