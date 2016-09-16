/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoreObjectFactory.h"

#include <itkObjectFactory.h>

#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkImageVtkMapper2D.h>
#include <mitkPointSet.h>
#include <mitkPointSetVtkMapper3D.h>
#include <mitkProperties.h>

#include "niftkCoordinateAxesVtkMapper3D.h"
#include "niftkFastPointSetVtkMapper3D.h"

namespace niftk
{

//-----------------------------------------------------------------------------
CoreObjectFactory::CoreObjectFactory()
: mitk::CoreObjectFactoryBase()
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    MITK_DEBUG << "niftk::CoreObjectFactory c'tor" << std::endl;

    MITK_DEBUG << "niftk::CoreObjectFactory c'tor finished" << std::endl;
  }
}


//-----------------------------------------------------------------------------
CoreObjectFactory::~CoreObjectFactory()
{
}


//-----------------------------------------------------------------------------
mitk::Mapper::Pointer CoreObjectFactory::CreateMapper(mitk::DataNode* node, MapperSlotId id)
{
  mitk::Mapper::Pointer newMapper = NULL;
  mitk::BaseData *data = node->GetData();

  if ( id == mitk::BaseRenderer::Standard3D )
  {
    if (dynamic_cast<mitk::PointSet*>(data) != NULL )
    {
      mitk::PointSet* pointSet = dynamic_cast<mitk::PointSet*>(data);
      if (pointSet->GetSize() > 1000)
      {
        newMapper = FastPointSetVtkMapper3D::New();
      }
      else
      {
        newMapper = mitk::PointSetVtkMapper3D::New();
      }
      newMapper->SetDataNode(node);
    }
    else if (dynamic_cast<CoordinateAxesData*>(data) != NULL)
    {
      newMapper = CoordinateAxesVtkMapper3D::New();
      newMapper->SetDataNode(node);
    }
  }
  return newMapper;
}


//-----------------------------------------------------------------------------
void CoreObjectFactory::SetDefaultProperties(mitk::DataNode* node)
{

  if(node == NULL)
  {
    return;
  }

  CoordinateAxesData::Pointer coordinateAxesData = dynamic_cast<CoordinateAxesData*>(node->GetData());
  if (coordinateAxesData.IsNotNull())
  {
    CoordinateAxesVtkMapper3D::SetDefaultProperties(node);
  }
}


//-----------------------------------------------------------------------------
void CoreObjectFactory::CreateFileExtensionsMap()
{
  MITK_DEBUG << "Registering additional file extensions." << std::endl;
  MITK_DEBUG << "Registering additional file extensions." << std::endl;
}


//-----------------------------------------------------------------------------
CoreObjectFactory::MultimapType CoreObjectFactory::GetFileExtensionsMap()
{
  return m_FileExtensionsMap;
}


//-----------------------------------------------------------------------------
CoreObjectFactory::MultimapType CoreObjectFactory::GetSaveFileExtensionsMap()
{
  return m_SaveFileExtensionsMap;
}


//-----------------------------------------------------------------------------
const char* CoreObjectFactory::GetFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_FileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}


//-----------------------------------------------------------------------------
const char* CoreObjectFactory::GetSaveFileExtensions()
{
  std::string fileExtension;
  this->CreateFileExtensions(m_SaveFileExtensionsMap, fileExtension);
  return fileExtension.c_str();
}


//-----------------------------------------------------------------------------
struct RegisterNifTKCoreObjectFactory{
  RegisterNifTKCoreObjectFactory()
    : m_Factory( CoreObjectFactory::New() )
  {
    MITK_DEBUG << "Registering niftk::CoreObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->RegisterExtraFactory( m_Factory );
  }

  ~RegisterNifTKCoreObjectFactory()
  {
    MITK_DEBUG << "Un-Registering niftk::CoreObjectFactory..." << std::endl;
    mitk::CoreObjectFactory::GetInstance()->UnRegisterExtraFactory( m_Factory );
  }

  CoreObjectFactory::Pointer m_Factory;
};


//-----------------------------------------------------------------------------
static RegisterNifTKCoreObjectFactory registerNifTKCoreObjectFactory;

}
