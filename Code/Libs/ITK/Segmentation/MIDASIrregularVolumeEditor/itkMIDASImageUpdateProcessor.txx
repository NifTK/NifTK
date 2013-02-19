/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkMIDASImageUpdateProcessor.h"

namespace itk
{

template<class TPixel, unsigned int VImageDimension>
MIDASImageUpdateProcessor<TPixel, VImageDimension>
::MIDASImageUpdateProcessor()
: m_DestinationImage(0)
{
  m_DestinationImage = ImageType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdateProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_DestinationImage=" << std::endl;
  os << indent.GetNextIndent() << m_DestinationImage << std::endl; 
}

template<class TPixel, unsigned int VImageDimension>
void 
MIDASImageUpdateProcessor<TPixel, VImageDimension>
::ValidateInputs()
{
  if (m_DestinationImage.IsNull())
  {
    itkExceptionMacro(<< "Destination image is NULL");
  }  
}

} // end namespace
