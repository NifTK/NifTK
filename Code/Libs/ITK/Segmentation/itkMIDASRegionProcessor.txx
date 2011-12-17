/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7491 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkMIDASRegionProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASRegionProcessor<TPixel, VImageDimension>
::MIDASRegionProcessor()
: m_Calculator(NULL)
, m_Processor(NULL)
{
  m_Calculator = CalculatorType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASRegionProcessor<TPixel, VImageDimension>
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
MIDASRegionProcessor<TPixel, VImageDimension>
::Undo()
{
  if (m_Processor.IsNull())
  {
    itkExceptionMacro(<< "Processor has not been set");
  }
  
  m_Processor->Undo();
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASRegionProcessor<TPixel, VImageDimension>
::Redo()
{
  if (m_Processor.IsNull())
  {
    itkExceptionMacro(<< "Processor has not been set");
  }
  
  m_Processor->Redo();
}

} // end namespace
