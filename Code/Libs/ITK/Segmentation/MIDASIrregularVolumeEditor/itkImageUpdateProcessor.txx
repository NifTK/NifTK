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
: m_DestinationImage(0)
{
  m_DestinationImage = ImageType::New();
}

template<class TPixel, unsigned int VImageDimension>
void 
ImageUpdateProcessor<TPixel, VImageDimension>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);  
  os << indent << "m_DestinationImage=" << std::endl;
  os << indent.GetNextIndent() << m_DestinationImage << std::endl; 
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
}

} // end namespace
