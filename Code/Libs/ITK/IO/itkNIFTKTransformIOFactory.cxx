/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
