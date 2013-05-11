/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkIOAdapter.h>
#include "mitkNifTKItkImageFileIOFactory.h"
#include "mitkNifTKItkImageFileReader.h"
#include <itkVersion.h>

namespace mitk
{
NifTKItkImageFileIOFactory::NifTKItkImageFileIOFactory()
{
  this->RegisterOverride("mitkIOAdapter",
                         "mitkNifTKItkImageFileReader",
                         "NifTK specific ITK based image IO",
                         1,
                         itk::CreateObjectFunction<IOAdapter<NifTKItkImageFileReader> >::New());
}

NifTKItkImageFileIOFactory::~NifTKItkImageFileIOFactory()
{
}

const char* NifTKItkImageFileIOFactory::GetITKSourceVersion() const
{
  return ITK_SOURCE_VERSION;
}

const char* NifTKItkImageFileIOFactory::GetDescription() const
{
  return "NifTKItkImageFile IO Factory, allows the NifTK specific loading of images supported by ITK";
}

} // end namespace mitk
