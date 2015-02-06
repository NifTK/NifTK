/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkNifTKImageIOFactory.h"
#include <itkCreateObjectFunction.h>
#include "itkAnalyzeImageIO.h"
#include "itkDRCAnalyzeImageIO.h"
#include "itkNiftiImageIO3201.h"
#include "itkINRImageIO.h"
#include "itkNifTKTransformIO.h"
#include <itkVersion.h>

namespace itk
{

NifTKImageIOFactory::NifTKImageIOFactory()
{
  this->RegisterOverride( "itkImageIOBase",
                          "itkAnalyzeImageIO",
                          "Analyze Image IO",
                          1,
                          CreateObjectFunction<AnalyzeImageIO>::New() );
  this->RegisterOverride("itkImageIOBase",
                         "itkDRCAnalyzeImageIO",
                         "DRC Analyze Image IO",
                         1,
                         CreateObjectFunction<DRCAnalyzeImageIO>::New());
  this->RegisterOverride("itkImageIOBase",
                         "itkNiftiImageIO3201",
                         "Nifti Image IO 3201",
                         1,
                         CreateObjectFunction<NiftiImageIO3201>::New());
  this->RegisterOverride("itkImageIOBase",
                         "itkINRImageIO",
                         "INR Image IO",
                         1,
                         CreateObjectFunction<INRImageIO>::New());
  this->RegisterOverride("itkTransformIOBase",
                         "itkNifTKTransformIO",
                         "Txt Transform IO",
                         1,
                         CreateObjectFunction<NifTKTransformIO>::New());
}

NifTKImageIOFactory::~NifTKImageIOFactory()
{
}

const char*
NifTKImageIOFactory::GetITKSourceVersion(void) const
{
  return ITK_SOURCE_VERSION;
}

const char*
NifTKImageIOFactory::GetDescription(void) const
{
  return "NifTK ImageIO Factory. Supports Analyze, DRC Analyze, NIfTI, INR and txt transform formats.";
}

} // end namespace itk

