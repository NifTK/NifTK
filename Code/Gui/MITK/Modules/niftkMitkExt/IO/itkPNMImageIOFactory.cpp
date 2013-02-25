/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkPNGImageIOFactory.cxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "itkPNMImageIOFactory.h"
#include <itkCreateObjectFunction.h>
#include "itkPNMImageIO.h"
#include <itkVersion.h>

namespace itk
{

PNMImageIOFactory::PNMImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "itkPNMImageIO",
                         "PNM Image IO",
                         1,
                         CreateObjectFunction<PNMImageIO>::New());
}
  
PNMImageIOFactory::~PNMImageIOFactory()
{
}

const char* 
PNMImageIOFactory::GetITKSourceVersion(void) const
{
  return ITK_SOURCE_VERSION;
}

const char* 
PNMImageIOFactory::GetDescription(void) const
{
  return "PNM ImageIO Factory, allows the loading of PNM images into insight";
}

} // end namespace itk
