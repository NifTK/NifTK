/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-27 16:30:27 +0100 (Wed, 27 Jul 2011) $
 Revision          : $Revision: 6863 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


#include "itkNIFTKTransformIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkNIFTKTransformIO.h"
#include "itkVersion.h"


namespace itk
{
void NIFTKTransformIOFactory::PrintSelf(std::ostream&, Indent) const
{

}


NIFTKTransformIOFactory::NIFTKTransformIOFactory()
{
  this->RegisterOverride("itkTransformIOBase",
                         "itkNIFTKTransformIO",
                         "Txt Transform IO",
                         1,
                         CreateObjectFunction<NIFTKTransformIO>::New());
}

NIFTKTransformIOFactory::~NIFTKTransformIOFactory()
{
}

const char*
NIFTKTransformIOFactory::GetITKSourceVersion(void) const
{
  return ITK_SOURCE_VERSION;
}

const char*
NIFTKTransformIOFactory::GetDescription() const
{
  return "Txt TransformIO Factory, allows the"
    " loading of Nifti images into insight";
}

} // end namespace itk
