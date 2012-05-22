/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: -1 $
 Last modified by  : $Author: $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkMIDASWipeProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASWipeProcessor<TPixel, VImageDimension>
::MIDASWipeProcessor()
{
  this->m_Processor = ClearProcessorType::New();
}

template<class TPixel, unsigned int VImageDimension>
void
MIDASWipeProcessor<TPixel, VImageDimension>
::SetWipeValue(const SegmentationPixelType value)
{
  ClearProcessorPointer clearProcessor = dynamic_cast<ClearProcessorType*>(this->m_Processor.GetPointer());
  if (clearProcessor.IsNotNull())
  {
    clearProcessor->SetWipeValue(value);
  }
  else
  {
    itkExceptionMacro(<< "The processor has not been set, or is the wrong class");
  }
}

template<class TPixel, unsigned int VImageDimension>
typename MIDASWipeProcessor<TPixel, VImageDimension>::SegmentationPixelType
MIDASWipeProcessor<TPixel, VImageDimension>
::GetWipeValue() const 
{
  ClearProcessorPointer clearProcessor = dynamic_cast<ClearProcessorType*>(this->m_Processor.GetPointer());
  if (clearProcessor.IsNotNull())
  {
    return clearProcessor->GetWipeValue();
  }
  else
  {
    itkExceptionMacro(<< "The processor has not been set, or is the wrong class");
  }  
}

} // end namespace
