/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : m.modat@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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

