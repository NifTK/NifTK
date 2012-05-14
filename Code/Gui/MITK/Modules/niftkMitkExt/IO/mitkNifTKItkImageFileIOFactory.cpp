/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-08 16:23:32 +0100 (Thu, 08 Sep 2011) $
 Revision          : $Revision: 7267 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkIOAdapter.h"
#include "mitkNifTKItkImageFileIOFactory.h"
#include "mitkNifTKItkImageFileReader.h"
#include "itkVersion.h"

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
