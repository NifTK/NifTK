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

#include "itkImageUpdatePasteRegionProcessor.h"
#include "itkImageRegionIterator.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
ImageUpdatePasteRegionProcessor<TPixel, VImageDimension>
::ImageUpdatePasteRegionProcessor()
: m_SourceImage(NULL)
, m_CopyBackground(true)
{
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdatePasteRegionProcessor<TPixel, VImageDimension>
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
ImageUpdatePasteRegionProcessor<TPixel, VImageDimension>
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
ImageUpdatePasteRegionProcessor<TPixel, VImageDimension> 
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
