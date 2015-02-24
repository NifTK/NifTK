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
#include "itkDRCAnalyzeImageIO.h"
#include "itkNiftiImageIO3201.h"
#include "itkINRImageIO.h"
#include "itkNifTKTransformIO.h"
#include <itkVersion.h>

#include <itkObjectFactory.h>
#include <niftkEnvironmentHelper.h>

namespace itk
{

NifTKImageIOFactory::NifTKImageIOFactory()
{
  /// Important note:
  ///
  /// Registering ITK image IOs to the ITK object factories here must follow the same
  /// logic as registering them to mitk::FileReaderRegistry in mitk::NifTKCoreObjectFactory
  /// in the niftkCore module.

  bool useDRCAnalyze = niftk::BooleanEnvironmentVariableIsOn("NIFTK_DRC_ANALYZE");

  if (useDRCAnalyze)
  {
    this->RegisterOverride("itkImageIOBase", "itkDRCAnalyzeImageIO", "DRC Analyze Image IO", 1,
       itk::CreateObjectFunction<DRCAnalyzeImageIO>::New());
  }

  this->RegisterOverride("itkImageIOBase", "itkNiftiImageIO3201", "Nifti Image IO 3201", 1,
    itk::CreateObjectFunction<NiftiImageIO3201>::New());

  this->RegisterOverride("itkImageIOBase", "itkINRImageIO", "INR Image IO", 1,
    itk::CreateObjectFunction<INRImageIO>::New());

  this->RegisterOverride("itkTransformIOBase", "itkNifTKTransformIO", "Txt Transform IO", 1,
    itk::CreateObjectFunction<NifTKTransformIO>::New());
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


//-----------------------------------------------------------------------------
struct RegisterNifTKImageIOFactory{
  RegisterNifTKImageIOFactory()
    : m_Factory( itk::NifTKImageIOFactory::New() )
  {
    itk::ObjectFactoryBase::RegisterFactory(m_Factory, itk::ObjectFactoryBase::INSERT_AT_FRONT);
  }

  ~RegisterNifTKImageIOFactory()
  {
    itk::ObjectFactoryBase::UnRegisterFactory(m_Factory);
  }

  itk::NifTKImageIOFactory::Pointer m_Factory;
};


//-----------------------------------------------------------------------------
static RegisterNifTKImageIOFactory registerNifTKImageIOFactory;

