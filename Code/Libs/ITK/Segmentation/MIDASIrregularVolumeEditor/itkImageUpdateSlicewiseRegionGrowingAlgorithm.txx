/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 05:46:51 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7494 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKIMAGEUPDATESLICEWISEREGIONGROWINGALGORITHM_TXX
#define ITKIMAGEUPDATESLICEWISEREGIONGROWINGALGORITHM_TXX

#include "itkImageUpdateSlicewiseRegionGrowingAlgorithm.h"

namespace itk
{

template<class TSegmentationPixel, class TGreyscalePixel, class TPointType, unsigned int VImageDimension>
ImageUpdateSlicewiseRegionGrowingAlgorithm<TSegmentationPixel, TGreyscalePixel, TPointType, VImageDimension>
::ImageUpdateSlicewiseRegionGrowingAlgorithm()
{
  m_RegionGrowingProcessor = RegionGrowingProcessorType::New();
}

template<class TSegmentationPixel, class TGreyscalePixel, class TPointType, unsigned int VImageDimension>
void 
ImageUpdateSlicewiseRegionGrowingAlgorithm<TSegmentationPixel, TGreyscalePixel, TPointType, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  if (m_RegionGrowingProcessor.IsNotNull())
  {
    os << indent << "m_RegionGrowingProcessor=" << m_RegionGrowingProcessor << std::endl;
  }
  else
  {
    os << indent << "m_RegionGrowingProcessor=NULL" << std::endl;
  }
}

template<class TSegmentationPixel, class TGreyscalePixel, class TPointType, unsigned int VImageDimension>
typename ImageUpdateSlicewiseRegionGrowingAlgorithm<TSegmentationPixel, TGreyscalePixel, TPointType, VImageDimension>::SegmentationImageType*
ImageUpdateSlicewiseRegionGrowingAlgorithm<TSegmentationPixel, TGreyscalePixel, TPointType, VImageDimension> 
::Execute(SegmentationImageType* imageToBeModified)
{

  m_RegionGrowingProcessor->SetDestinationImage(imageToBeModified);
  m_RegionGrowingProcessor->Execute();
  
  return m_RegionGrowingProcessor->GetDestinationImage();
}

} // end namespace

#endif
