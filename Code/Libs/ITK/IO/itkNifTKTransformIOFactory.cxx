/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkNifTKTransformIOFactory.h"
#include <itkCreateObjectFunction.h>
#include "itkNifTKTransformIO.h"
#include <itkVersion.h>


namespace itk
{
void NifTKTransformIOFactory::PrintSelf(std::ostream&, Indent) const
{

}


NifTKTransformIOFactory::NifTKTransformIOFactory()
{
  this->RegisterOverride("itkTransformIOBase",
                         "itkNifTKTransformIO",
                         "Txt Transform IO",
                         1,
                         CreateObjectFunction<NifTKTransformIO>::New());
}

NifTKTransformIOFactory::~NifTKTransformIOFactory()
{
}

const char*
NifTKTransformIOFactory::GetITKSourceVersion(void) const
{
  return ITK_SOURCE_VERSION;
}

const char*
NifTKTransformIOFactory::GetDescription() const
{
  return "Txt TransformIO Factory, allows the"
    " loading of Nifti images into insight";
}

} // end namespace itk
