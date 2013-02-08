/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkINRImageIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkINRImageIO.h"
#include "itkVersion.h"

namespace itk
{

INRImageIOFactory::INRImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "itkINRImageIO",
                         "INR Image IO",
                         1,
                         CreateObjectFunction<INRImageIO>::New());
}

INRImageIOFactory::~INRImageIOFactory()
{
}

const char*
INRImageIOFactory::GetITKSourceVersion(void) const
{
  return ITK_SOURCE_VERSION;
}

const char*
INRImageIOFactory::GetDescription(void) const
{
  return "INR ImageIO Factory, allows the loading of INR images into insight";
}

} // end namespace itk

