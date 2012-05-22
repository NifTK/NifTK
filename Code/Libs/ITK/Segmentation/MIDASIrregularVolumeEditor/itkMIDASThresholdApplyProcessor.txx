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

#include "itkMIDASThresholdApplyProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASThresholdApplyProcessor<TPixel, VImageDimension>
::MIDASThresholdApplyProcessor()
: m_CopyProcessor(NULL)
, m_ClearProcessor(NULL)
, m_Calculator(NULL)
{
  m_Calculator = CalculatorType::New();
  m_CopyProcessor = CopyProcessorType::New();
  m_ClearProcessor = ClearProcessorType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASThresholdApplyProcessor<TPixel, VImageDimension>
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
  if (m_CopyProcessor.IsNotNull())
  {
    os << indent << "m_CopyProcessor=" << m_CopyProcessor << std::endl;
  }
  else
  {
    os << indent << "m_CopyProcessor=NULL" << std::endl;
  }
  if (m_ClearProcessor.IsNotNull())
  {
    os << indent << "m_ClearProcessor=" << m_ClearProcessor << std::endl;
  }
  else
  {
    os << indent << "m_ClearProcessor=NULL" << std::endl;
  }
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASThresholdApplyProcessor<TPixel, VImageDimension>
::CalculateRegionOfInterest()
{
  RegionType roi = this->m_Calculator->GetMinimumRegion(this->GetSourceImage(), 0);
  this->m_CopyProcessor->SetSourceRegionOfInterest(roi);
  this->m_CopyProcessor->SetDestinationRegionOfInterest(roi);
  this->m_ClearProcessor->SetDestinationRegionOfInterest(roi);
  this->Modified();
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASThresholdApplyProcessor<TPixel, VImageDimension>
::Undo()
{
  this->m_ClearProcessor->SetDestinationImage(this->GetSourceImage());
  this->m_ClearProcessor->Undo();
  this->SetSourceImage(this->m_ClearProcessor->GetDestinationImage());
  
  this->m_CopyProcessor->SetDestinationImage(this->GetDestinationImage());
  this->m_CopyProcessor->Undo();
  this->SetDestinationImage(this->m_CopyProcessor->GetDestinationImage());
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASThresholdApplyProcessor<TPixel, VImageDimension>
::Redo()
{
  this->m_CopyProcessor->SetSourceImage(this->GetSourceImage());
  this->m_CopyProcessor->SetDestinationImage(this->GetDestinationImage());
  this->m_CopyProcessor->Redo();
  this->SetDestinationImage(this->m_CopyProcessor->GetDestinationImage());
  
  this->m_ClearProcessor->SetDestinationImage(this->GetSourceImage());
  this->m_ClearProcessor->SetWipeValue(0);  
  this->m_ClearProcessor->Redo();
  this->SetSourceImage(this->m_ClearProcessor->GetDestinationImage());  
}

} // end namespace
