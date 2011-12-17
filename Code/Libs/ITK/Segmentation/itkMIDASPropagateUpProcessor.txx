/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7442 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkMIDASPropagateUpProcessor.h"

namespace itk
{

template<class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
MIDASPropagateUpProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>
::MIDASPropagateUpProcessor()
{
}

template<class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
void
MIDASPropagateUpProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>
::SetOrientationAndSlice(itk::ORIENTATION_ENUM orientation, int sliceNumber)
{
  MIDASPropagateProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>::SetOrientationAndSlice(orientation, sliceNumber);
  
  RegionType region = this->m_Calculator->GetPlusOrUpRegion(this->GetDestinationImage(), orientation, sliceNumber);
  this->m_Strategy->SetDestinationImage(this->GetDestinationImage());
  this->m_Strategy->SetDestinationRegionOfInterest(region);
  this->m_Algorithm->SetRegionOfInterest(region);
  this->m_Algorithm->SetSliceNumber(sliceNumber);
  this->m_Algorithm->SetOrientation(orientation);
  this->Modified();
}


} // end namespace
