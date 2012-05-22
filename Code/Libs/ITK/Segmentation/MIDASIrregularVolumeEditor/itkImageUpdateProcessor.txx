/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-15 07:06:41 +0100 (Sat, 15 Oct 2011) $
 Revision          : $Revision: 7522 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkImageUpdateProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
ImageUpdateProcessor<TPixel, VImageDimension>
::ImageUpdateProcessor()
: m_UpdateCalculated(false)
, m_DestinationImage(0)
, m_BeforeImage(0)
, m_AfterImage(0)
{
  SizeType size;
  size.Fill(0);
  IndexType voxelIndex;
  voxelIndex.Fill(0);
  RegionType region;
  region.SetSize(size);
  region.SetIndex(voxelIndex);
  
  m_DestinationRegionOfInterest = region;
  
  m_DestinationImage = ImageType::New();
  m_BeforeImage = ImageType::New();
  m_AfterImage = ImageType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdateProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_UpdateCalculated=" << m_UpdateCalculated << std::endl;
  os << indent << "m_DestinationRegionOfInterest=\n" << m_DestinationRegionOfInterest << std::endl;
  os << indent << "m_DestinationImage=" << std::endl;
  os << indent.GetNextIndent() << m_DestinationImage << std::endl; 
  os << indent << "m_BeforeImage=" << std::endl;
  os << indent.GetNextIndent() << m_BeforeImage << std::endl; 
  os << indent << "m_AfterImage=" << std::endl;
  os << indent.GetNextIndent() << m_AfterImage << std::endl; 
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdateProcessor<TPixel, VImageDimension>
::SetDestinationRegionOfInterest(std::vector<int> &array)
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
  
  this->SetDestinationRegionOfInterest(region);
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdateProcessor<TPixel, VImageDimension>
::ValidateInputs()
{
  if (m_DestinationImage.IsNull())
  {
    itkExceptionMacro(<< "Destination image is NULL");
  }
  
  SizeType size = m_DestinationRegionOfInterest.GetSize();
  unsigned long int numberOfVoxels = size[0];
  
  for (unsigned int i = 1; i < VImageDimension; i++)
  {
    numberOfVoxels *= size[i];
  }
  
  if (numberOfVoxels == 0)
  {
    itkExceptionMacro(<< "Destination region of interest has zero size");
  }
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdateProcessor<TPixel, VImageDimension>
::CopyImageRegionToDestination(ImagePointer sourceImage)
{
  // Here we assume:
  // sourceImage has a LargestPossibleRegion that is fully contained within the destinationImage
  // and the size and index directly correspond to the correct region you want pasting.
  
  PasteImageFilterPointer imagePasteFilter = PasteImageFilterType::New();
  imagePasteFilter->InPlaceOn();
  imagePasteFilter->SetSourceImage(sourceImage);
  imagePasteFilter->SetSourceRegion(sourceImage->GetLargestPossibleRegion()); 
  imagePasteFilter->SetDestinationImage(m_DestinationImage);
  imagePasteFilter->SetDestinationIndex(sourceImage->GetLargestPossibleRegion().GetIndex()); 
  imagePasteFilter->Update();
  
  // The memory address changes due to InPlaceOn.
  m_DestinationImage = imagePasteFilter->GetOutput();
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdateProcessor<TPixel, VImageDimension>
::Redo()
{
  this->ValidateInputs();

  itkDebugMacro( << "ImageUpdateProcessor::Destination image is at address=" << m_DestinationImage.GetPointer() \
    << ", and has region=\n" << m_DestinationImage->GetLargestPossibleRegion() \
    << ", and we process ROI=\n" << m_DestinationRegionOfInterest \
    );
  
  if (!m_UpdateCalculated)
  {
    ExtractImageFilterPointer extractImageFilter = ExtractImageFilterType::New();
    extractImageFilter->SetInput(m_DestinationImage);
    extractImageFilter->SetExtractionRegion(m_DestinationRegionOfInterest);
    extractImageFilter->Update();
    
    m_BeforeImage = extractImageFilter->GetOutput();
    m_BeforeImage->DisconnectPipeline();
    
    extractImageFilter->Update();
    m_AfterImage = extractImageFilter->GetOutput();
    m_AfterImage->DisconnectPipeline();
    
    itkDebugMacro( << "ImageUpdateProcessor::m_BeforeImage has address=" << m_BeforeImage.GetPointer() \
      << ", m_BeforeImage has region=\n" << m_BeforeImage->GetLargestPossibleRegion() \
      << ", m_AfterImage has address=" << m_AfterImage.GetPointer() \
      << ", m_AfterImage has region=\n" << m_AfterImage->GetLargestPossibleRegion() \
      );
      
    // Let derived classes make changes to the after image.
    this->ApplyUpdateToAfterImage();    
  }

  itkDebugMacro( << "Copying m_AfterImage to m_DestinationImage - started");
  
  this->CopyImageRegionToDestination(m_AfterImage);
  
  itkDebugMacro( << "Copying m_AfterImage to m_DestinationImage - finished");
  m_UpdateCalculated = true; 
}

template<class TPixel, unsigned int VImageDimension>
void
ImageUpdateProcessor<TPixel, VImageDimension>
::Undo()
{
  this->ValidateInputs();
  this->CopyImageRegionToDestination(m_BeforeImage);
}

} // end namespace
