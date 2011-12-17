/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-12 16:43:10 +0100 (Mon, 12 Sep 2011) $
 Revision          : $Revision: 7299 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkDRCCoreObjectFactory.h"

#include "mitkProperties.h"
#include "mitkBaseRenderer.h"
#include "mitkDataNode.h"
#include "mitkImage.h"
#include "mitkDRCItkImageFileIOFactory.h"
#include "mitkVolumeDataVtkMapper3D.h"
#include "mitkImageVtkMapper2D.h"
#include "mitkItkImageFileIOFactory.h"

#include "itkObjectFactory.h"

mitk::DRCCoreObjectFactory::DRCCoreObjectFactory(bool /*registerSelf*/)
:CoreObjectFactoryBase()
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    MITK_INFO << "DRCCoreObjectFactory c'tor" << std::endl;

    // At this point in this constructor, the main MITK CoreObjectFactory has been created,
    // (because in RegisterDRCCoreObjectFactory, the call to mitk::CoreObjectFactory::GetInstance()
    // will instantiate the MITK CoreObjectFactory, which will create lots of Core MITK objects),
    // so MITKs file reader for ITK images will already be available. So, now we remove it.
    std::list<itk::ObjectFactoryBase*> listOfObjectFactories = itk::ObjectFactoryBase::GetRegisteredFactories();
    std::list<itk::ObjectFactoryBase*>::iterator iter;
    mitk::ItkImageFileIOFactory::Pointer itkIOFactory = NULL;
    for (iter = listOfObjectFactories.begin(); iter != listOfObjectFactories.end(); iter++)
    {
      itkIOFactory = dynamic_cast<mitk::ItkImageFileIOFactory*>(*iter);
      if (itkIOFactory.IsNotNull())
      {
        break;
      }
    }
    itk::ObjectFactoryBase::UnRegisterFactory(itkIOFactory.GetPointer());

    // Load our specific factory, which will be used to load all ITK images, just like the MITK one,
    // but then in addition, will load DRC Analyze files differently.
    mitk::DRCItkImageFileIOFactory::RegisterOneFactory();

    // Carry on as per normal.
    CreateFileExtensionsMap();
    alreadyDone = true;
    MITK_INFO << "DRCCoreObjectFactory c'tor finished" << std::endl;
  }

}

mitk::Mapper::Pointer mitk::DRCCoreObjectFactory::CreateMapper(mitk::DataNode* node, MapperSlotId id)
{
  mitk::Mapper::Pointer newMapper = NULL;
  mitk::BaseData *data = node->GetData();

  if ( id == mitk::BaseRenderer::Standard3D )
  {
    if((dynamic_cast<Image*>(data) != NULL))
    {
      newMapper = mitk::VolumeDataVtkMapper3D::New();
      newMapper->SetDataNode(node);
    }
  }
  return newMapper;
}

void mitk::DRCCoreObjectFactory::SetDefaultProperties(mitk::DataNode* node)
{

  if(node == NULL)
  {
    return;
  }

  mitk::DataNode::Pointer nodePointer = node;

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
  if(image.IsNotNull() && image->IsInitialized())
  {
    mitk::ImageVtkMapper2D::SetDefaultProperties(node);
    mitk::VolumeDataVtkMapper3D::SetDefaultProperties(node);
  }
}

void mitk::DRCCoreObjectFactory::CreateFileExtensionsMap()
{
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.img", "Dementia Research Centre Analyze Image"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.img.gz", "Dementia Research Centre Compressed Analyze Image"));

  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>("*.img", "Dementia Research Centre Analyze Image"));
  m_SaveFileExtensionsMap.insert(std::pair<std::string, std::string>("*.img.gz", "Dementia Research Centre Compressed Analyze Image"));
}

mitk::DRCCoreObjectFactory::MultimapType mitk::DRCCoreObjectFactory::GetFileExtensionsMap()
{
  return m_FileExtensionsMap;
}

mitk::DRCCoreObjectFactory::MultimapType mitk::DRCCoreObjectFactory::GetSaveFileExtensionsMap()
{
  return m_SaveFileExtensionsMap;
}

const char* mitk::DRCCoreObjectFactory::GetFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_FileExtensionsMap, fileExtension);
  return fileExtension.c_str();
};

const char* mitk::DRCCoreObjectFactory::GetSaveFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_SaveFileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}

void RegisterDRCCoreObjectFactory()
{
  static bool oneDRCCoreObjectFactoryRegistered = false;
  if ( ! oneDRCCoreObjectFactoryRegistered )
  {
    MITK_INFO << "Registering DRCCoreObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->RegisterExtraFactory(mitk::DRCCoreObjectFactory::New());
    oneDRCCoreObjectFactoryRegistered = true;
  }
}
