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
#include <mitkProperties.h>
#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkPointSet.h>
#include <mitkCoordinateAxesVtkMapper3D.h>
#include <mitkVolumeDataVtkMapper3D.h>
#include <mitkImageVtkMapper2D.h>
#include <mitkFastPointSetVtkMapper3D.h>
#include <mitkPointSetVtkMapper3D.h>

//-----------------------------------------------------------------------------
mitk::NifTKCoreObjectFactory::NifTKCoreObjectFactory()
:CoreObjectFactoryBase()
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    MITK_DEBUG << "NifTKCoreObjectFactory c'tor" << std::endl;

    MITK_DEBUG << "NifTKCoreObjectFactory c'tor finished" << std::endl;
  }
}


//-----------------------------------------------------------------------------
mitk::NifTKCoreObjectFactory::~NifTKCoreObjectFactory()
{
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
  MITK_DEBUG << "Registering additional file extensions." << std::endl;
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

