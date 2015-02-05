/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkDRCAnalyzeImageIOFactory.h"
#include "itkDRCAnalyzeImageIO.h"
#include <itkVersion.h>

namespace itk
{

DRCAnalyzeImageIOFactory::DRCAnalyzeImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "itkDRCAnalyzeImageIO",
                         "DRC Analyze Image IO",
                         1,
                         CreateObjectFunction<DRCAnalyzeImageIO>::New());
}

DRCAnalyzeImageIOFactory::~DRCAnalyzeImageIOFactory()
{
}

const char *
DRCAnalyzeImageIOFactory::GetITKSourceVersion(void) const
{
  return ITK_SOURCE_VERSION;
}

const char *
DRCAnalyzeImageIOFactory::GetDescription() const
{
  return "DRC Analyze ImageIO Factory, allows the loading of DRC Analyze images into insight";
}

} // end namespace itk
