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
#include "mitkDRCItkImageFileIOFactory.h"
#include "mitkDRCItkImageFileReader.h"
#include "itkVersion.h"

namespace mitk
{
DRCItkImageFileIOFactory::DRCItkImageFileIOFactory()
{
  this->RegisterOverride("mitkIOAdapter",
                         "mitkDRCItkImageFileReader",
                         "DRC specific ITK based image IO",
                         1,
                         itk::CreateObjectFunction<IOAdapter<DRCItkImageFileReader> >::New());
}

DRCItkImageFileIOFactory::~DRCItkImageFileIOFactory()
{
}

const char* DRCItkImageFileIOFactory::GetITKSourceVersion() const
{
  return ITK_SOURCE_VERSION;
}

const char* DRCItkImageFileIOFactory::GetDescription() const
{
  return "DRCItkImageFile IO Factory, allows the DRC specific loading of images supported by ITK";
}

} // end namespace mitk
