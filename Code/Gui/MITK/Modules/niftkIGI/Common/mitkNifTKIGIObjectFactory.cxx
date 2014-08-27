/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkNifTKIGIObjectFactory.h"

#include <itkObjectFactory.h>
#include <mitkProperties.h>
#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include "mitkImage2DToTexturePlaneMapper3D.h"

//-----------------------------------------------------------------------------
mitk::NifTKIGIObjectFactory::NifTKIGIObjectFactory()
:CoreObjectFactoryBase()
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    CreateFileExtensionsMap();
    alreadyDone = true;
  }
}


//-----------------------------------------------------------------------------
mitk::Mapper::Pointer mitk::NifTKIGIObjectFactory::CreateMapper(mitk::DataNode* node, MapperSlotId id)
{
  mitk::Mapper::Pointer newMapper = NULL;
  mitk::BaseData *data = node->GetData();

  if ( id == mitk::BaseRenderer::Standard3D )
  {
    /* This would get added to ALL images. For IGI TrackedImageView we only want the one being tracked.
    if (dynamic_cast<mitk::Image*>(data) != NULL)
    {
      newMapper = mitk::Image2DToTexturePlaneMapper3D::New();
      newMapper->SetDataNode(node);
    }
    */
  }
  return newMapper;
}


//-----------------------------------------------------------------------------
void mitk::NifTKIGIObjectFactory::SetDefaultProperties(mitk::DataNode* node)
{
  if(node == NULL)
  {
    return;
  }

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
  if (image.IsNotNull())
  {
    mitk::Image2DToTexturePlaneMapper3D::SetDefaultProperties(node);
  }
}


//-----------------------------------------------------------------------------
void mitk::NifTKIGIObjectFactory::CreateFileExtensionsMap()
{
}


//-----------------------------------------------------------------------------
mitk::NifTKIGIObjectFactory::MultimapType mitk::NifTKIGIObjectFactory::GetFileExtensionsMap()
{
  return m_FileExtensionsMap;
}


//-----------------------------------------------------------------------------
mitk::NifTKIGIObjectFactory::MultimapType mitk::NifTKIGIObjectFactory::GetSaveFileExtensionsMap()
{
  return m_SaveFileExtensionsMap;
}


//-----------------------------------------------------------------------------
const char* mitk::NifTKIGIObjectFactory::GetFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_FileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}


//-----------------------------------------------------------------------------
const char* mitk::NifTKIGIObjectFactory::GetSaveFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_SaveFileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}


//-----------------------------------------------------------------------------
struct RegisterNifTKIGIObjectFactory{
  RegisterNifTKIGIObjectFactory()
    : m_Factory( mitk::NifTKIGIObjectFactory::New() )
  {
    MITK_DEBUG << "Registering NifTKIGIObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->RegisterExtraFactory( m_Factory );
  }

  ~RegisterNifTKIGIObjectFactory()
  {
    MITK_DEBUG << "Un-Registering NifTKIGIObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->UnRegisterExtraFactory( m_Factory );
  }

  mitk::NifTKIGIObjectFactory::Pointer m_Factory;
};


//-----------------------------------------------------------------------------
static RegisterNifTKIGIObjectFactory registerNifTKIGIObjectFactory;

