/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASRetainMarksNoThresholdingProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASRetainMarksNoThresholdingProcessor<TPixel, VImageDimension>
::MIDASRetainMarksNoThresholdingProcessor()
: m_Processor(NULL)
, m_Calculator(NULL)
{
  m_Calculator = CalculatorType::New();
  m_Processor = ProcessorType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASRetainMarksNoThresholdingProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  if (m_Calculator.IsNotNull())
  {
    os << indent << "m_Calculator=" << m_Calculator << std::endl;
  }
  else
  {
    os << indent << "m_Calculator=NULL" << std::endl;
  }
  if (m_Processor.IsNotNull())
  {
    os << indent << "m_Processor=" << m_Processor << std::endl;
  }
  else
  {
    os << indent << "m_Processor=NULL" << std::endl;
  }
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASRetainMarksNoThresholdingProcessor<TPixel, VImageDimension>
::SetSlices(itk::ORIENTATION_ENUM orientation, 
  int sourceSliceNumber, int targetSliceNumber)
{
  RegionType sourceRegion = this->m_Calculator->GetSliceRegion(this->GetSourceImage(), orientation, sourceSliceNumber);
  RegionType targetRegion = this->m_Calculator->GetSliceRegion(this->GetDestinationImage(), orientation, targetSliceNumber);
  this->m_Processor->SetSourceRegionOfInterest(sourceRegion);
  this->m_Processor->SetDestinationRegionOfInterest(targetRegion);
  this->Modified();
}

} // end namespace
