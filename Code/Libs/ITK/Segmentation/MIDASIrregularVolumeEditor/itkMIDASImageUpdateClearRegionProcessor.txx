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

#include "itkMIDASImageUpdateClearRegionProcessor.h"
#include "itkImageRegionIterator.h"

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
