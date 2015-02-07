/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkNifTKCoreObjectFactory.h"

#include <itkObjectFactory.h>

#include <mitkAbstractFileIO.h>
#include <mitkItkImageIO_p.h>
#include <mitkProperties.h>
#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkIOMimeTypes.h>
#include <mitkPointSet.h>
#include <mitkCoordinateAxesData.h>
#include <mitkCoordinateAxesDataWriter.h>
#include <mitkCoordinateAxesDataReaderFactory.h>
#include <mitkCoordinateAxesDataWriterFactory.h>
#include <mitkCoordinateAxesVtkMapper3D.h>
#include <mitkVolumeDataVtkMapper3D.h>
#include <mitkImageVtkMapper2D.h>
#include <itkPNMImageIOFactory.h>
#include <mitkFastPointSetVtkMapper3D.h>
#include <mitkPointSetVtkMapper3D.h>

#include <niftkEnvironmentHelper.h>
#include <itkAnalyzeImageIO.h>
#include <itkDRCAnalyzeImageIO.h>
#include <itkINRImageIO.h>
#include <itkNiftiImageIO3201.h>

//-----------------------------------------------------------------------------
mitk::NifTKCoreObjectFactory::NifTKCoreObjectFactory()
:CoreObjectFactoryBase()
, m_PNMImageIOFactory(itk::PNMImageIOFactory::New().GetPointer())
, m_CoordinateAxesDataReaderFactory(mitk::CoordinateAxesDataReaderFactory::New().GetPointer())
, m_CoordinateAxesDataWriterFactory(mitk::CoordinateAxesDataWriterFactory::New().GetPointer())
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    MITK_DEBUG << "NifTKCoreObjectFactory c'tor" << std::endl;

    /// Important note:
    ///
    /// Registering ITK image IOs to mitk::FileReaderRegistry here must follow the same
    /// logic as registering them to the ITK object factories in itk::NifTKImageIOFactory
    /// in the niftkITK library.

    /// TODO
    /// ITK readers and writers should be registered from itk::NifTKImageIOFactory.
    itk::ObjectFactoryBase::RegisterFactory(m_PNMImageIOFactory);
    itk::ObjectFactoryBase::RegisterFactory(m_CoordinateAxesDataReaderFactory);
    itk::ObjectFactoryBase::RegisterFactory(m_CoordinateAxesDataWriterFactory);

    /// TODO
    /// Do not access m_FileWriters. Use mitk::WriterRegistry instead.
    m_FileWriters.push_back(mitk::CoordinateAxesDataWriter::New().GetPointer());

    bool useDRCAnalyze = niftk::BooleanEnvironmentVariableIsOn("NIFTK_DRC_ANALYZE");

    if (useDRCAnalyze)
    {
      itk::DRCAnalyzeImageIO::Pointer itkDrcAnalyzeIO = itk::DRCAnalyzeImageIO::New();
      mitk::ItkImageIO* drcAnalyzeIO = new mitk::ItkImageIO(mitk::IOMimeTypes::NIFTI_MIMETYPE(), itkDrcAnalyzeIO.GetPointer(), 2);
      m_FileIOs.push_back(drcAnalyzeIO);
    }
    else
    {
      itk::AnalyzeImageIO::Pointer itkAnalyzeIO = itk::AnalyzeImageIO::New();
      mitk::ItkImageIO* analyzeIO = new mitk::ItkImageIO(mitk::IOMimeTypes::NIFTI_MIMETYPE(), itkAnalyzeIO.GetPointer(), 2);
      m_FileIOs.push_back(analyzeIO);
    }

    itk::NiftiImageIO3201::Pointer itkNiftiIO = itk::NiftiImageIO3201::New();
    mitk::ItkImageIO* niftiIO = new mitk::ItkImageIO(mitk::IOMimeTypes::NIFTI_MIMETYPE(), itkNiftiIO.GetPointer(), 1);
    m_FileIOs.push_back(niftiIO);

    itk::INRImageIO::Pointer itkINRImageIO = itk::INRImageIO::New();
    mitk::ItkImageIO* inrImageIO = new mitk::ItkImageIO(Self::INRIA_MIMETYPE(), itkINRImageIO.GetPointer(), 0);
    m_FileIOs.push_back(inrImageIO);

    CreateFileExtensionsMap();
    alreadyDone = true;

    MITK_DEBUG << "NifTKCoreObjectFactory c'tor finished" << std::endl;
  }

}


//-----------------------------------------------------------------------------
mitk::NifTKCoreObjectFactory::~NifTKCoreObjectFactory()
{
  /// TODO
  /// ITK readers and writers should be unregistered from itk::NifTKImageIOFactory.
  itk::ObjectFactoryBase::UnRegisterFactory(m_PNMImageIOFactory);
  itk::ObjectFactoryBase::UnRegisterFactory(m_CoordinateAxesDataReaderFactory);
  itk::ObjectFactoryBase::UnRegisterFactory(m_CoordinateAxesDataWriterFactory);

  for(std::vector<mitk::AbstractFileIO*>::iterator iter = m_FileIOs.begin(),
      endIter = m_FileIOs.end(); iter != endIter; ++iter)
  {
    delete *iter;
  }
}

//-----------------------------------------------------------------------------
mitk::Mapper::Pointer mitk::NifTKCoreObjectFactory::CreateMapper(mitk::DataNode* node, MapperSlotId id)
{
  mitk::Mapper::Pointer newMapper = NULL;
  mitk::BaseData *data = node->GetData();

  if ( id == mitk::BaseRenderer::Standard3D )
  {
    if (dynamic_cast<PointSet*>(data) != NULL )
    {
      mitk::PointSet* pointSet = dynamic_cast<PointSet*>(data);
      if (pointSet->GetSize() > 1000)
      {
        newMapper = mitk::FastPointSetVtkMapper3D::New();
      }
      else
      {
        newMapper = mitk::PointSetVtkMapper3D::New();
      }
      newMapper->SetDataNode(node);
    }
    else if (dynamic_cast<CoordinateAxesData*>(data) != NULL)
    {
      newMapper = mitk::CoordinateAxesVtkMapper3D::New();
      newMapper->SetDataNode(node);
    }
  }
  return newMapper;
}


//-----------------------------------------------------------------------------
void mitk::NifTKCoreObjectFactory::SetDefaultProperties(mitk::DataNode* node)
{

  if(node == NULL)
  {
    return;
  }

  mitk::CoordinateAxesData::Pointer coordinateAxesData = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
  if (coordinateAxesData.IsNotNull())
  {
    mitk::CoordinateAxesVtkMapper3D::SetDefaultProperties(node);
  }
}


//-----------------------------------------------------------------------------
void mitk::NifTKCoreObjectFactory::CreateFileExtensionsMap()
{
  MITK_DEBUG << "Registering additional file extensions." << std::endl;

  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.pgm", "Portable Gray Map"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.ppm", "Portable Pixel Map"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.pbm", "Portable Binary Map"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.pnm", "Portable aNy Map"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>(mitk::CoordinateAxesData::FILE_EXTENSION_WITH_ASTERISK, mitk::CoordinateAxesData::FILE_DIALOG_NAME));

  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>("*.pgm", "Portable Gray Map"));
  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>("*.ppm", "Portable Pixel Map"));
  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>("*.pbm", "Portable Binary Map"));
  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>("*.pnm", "Portable aNy Map"));
  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>(mitk::CoordinateAxesData::FILE_EXTENSION_WITH_ASTERISK, mitk::CoordinateAxesData::FILE_DIALOG_NAME));
}


//-----------------------------------------------------------------------------
mitk::NifTKCoreObjectFactory::MultimapType mitk::NifTKCoreObjectFactory::GetFileExtensionsMap()
{
  return m_FileExtensionsMap;
}


//-----------------------------------------------------------------------------
mitk::NifTKCoreObjectFactory::MultimapType mitk::NifTKCoreObjectFactory::GetSaveFileExtensionsMap()
{
  return m_SaveFileExtensionsMap;
}


//-----------------------------------------------------------------------------
const char* mitk::NifTKCoreObjectFactory::GetFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_FileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}


//-----------------------------------------------------------------------------
const char* mitk::NifTKCoreObjectFactory::GetSaveFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_SaveFileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType mitk::NifTKCoreObjectFactory::INRIA_MIMETYPE()
{
  CustomMimeType mimeType(Self::INRIA_MIMETYPE_NAME());
  mimeType.AddExtension("inr");
  mimeType.AddExtension("inr.gz");
  mimeType.SetCategory("Images");
  mimeType.SetComment("INRIA image");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string mitk::NifTKCoreObjectFactory::INRIA_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".image.inria";
  return name;
}


//-----------------------------------------------------------------------------
struct RegisterNifTKCoreObjectFactory{
  RegisterNifTKCoreObjectFactory()
    : m_Factory( mitk::NifTKCoreObjectFactory::New() )
  {
    MITK_DEBUG << "Registering NifTKCoreObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->RegisterExtraFactory( m_Factory );
  }

  ~RegisterNifTKCoreObjectFactory()
  {
    MITK_DEBUG << "Un-Registering NifTKCoreObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->UnRegisterExtraFactory( m_Factory );
  }

  mitk::NifTKCoreObjectFactory::Pointer m_Factory;
};


//-----------------------------------------------------------------------------
static RegisterNifTKCoreObjectFactory registerNifTKCoreObjectFactory;

