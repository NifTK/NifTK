/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIOUtil.h"

#include <QFileInfo>
#include <QXmlSimpleReader>

#include <mitkIOUtil.h>

#include <usGetModuleContext.h>
#include <usModuleContext.h>
#include <usServiceReference.h>

#include <niftkLookupTableContainer.h>
#include <niftkLookupTableProviderService.h>
#include <niftkLookupTableSaxHandler.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IOUtil::IOUtil()
{
}


//-----------------------------------------------------------------------------
IOUtil::~IOUtil()
{
}


//-----------------------------------------------------------------------------
QString IOUtil::LoadLookupTable(QString& fileName)
{
  QString lutName;

  QFileInfo finfo(fileName);
  if (!finfo.exists())
  {
    return lutName;
  }

  // create a lookup table
  LookupTableProviderService* lutService = Self::GetLookupTableProviderService();
  LookupTableContainer * loadedContainer;

  if (fileName.contains(".lut"))
  {
    QFile file(fileName);
    QXmlInputSource inputSource(&file);

    QXmlSimpleReader reader;
    LookupTableSaxHandler handler;
    reader.setContentHandler(&handler);
    reader.setErrorHandler(&handler);

    if (reader.parse(inputSource))
    {
      loadedContainer = handler.GetLookupTableContainer();
    }
    else
    {
      MITK_ERROR << "niftk::LookupTableManager(): failed to parse XML file (" << fileName.toStdString()
                 << ") so returning null";
    }
  }
  else
  {
    std::vector<mitk::BaseData::Pointer> containerData = mitk::IOUtil::Load(fileName.toStdString());
    if (containerData.empty())
    {
      MITK_ERROR << "Unable to load LookupTableContainer from " << fileName.toStdString();
    }
    else
    {
      loadedContainer =
        dynamic_cast<LookupTableContainer* >(containerData.at(0).GetPointer());

      if (loadedContainer != NULL)
      {
        loadedContainer->SetDisplayName(loadedContainer->GetDisplayName());
        loadedContainer->SetOrder(lutService->GetNumberOfLookupTables());
      }
    }
  }

  if (loadedContainer != NULL)
  {
    lutService->AddNewLookupTableContainer(loadedContainer);
    lutName = loadedContainer->GetDisplayName();
  }

  return lutName;
}


//-----------------------------------------------------------------------------
LookupTableProviderService* IOUtil::GetLookupTableProviderService()
{
  us::ModuleContext* context = us::GetModuleContext();
  us::ServiceReference<LookupTableProviderService> serviceRef = context->GetServiceReference<LookupTableProviderService>();
  LookupTableProviderService* lutService = context->GetService<LookupTableProviderService>(serviceRef);

  if (lutService == nullptr)
  {
    mitkThrow() << "Failed to find niftk::LookupTableProviderService." << std::endl;
  }

  return lutService;
}

}
