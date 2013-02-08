/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKDRCANALYZEIMAGEIO_CXX
#define ITKDRCANALYZEIMAGEIO_CXX

#include "itkDRCAnalyzeImageIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkDRCAnalyzeImageIO.h"
#include "itkVersion.h"

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

} // end namespace itk

#endif
