/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPluginActivator.h"

#include <berryPlatform.h>
#include <berryPlatformUI.h>
#include <berryIWorkbenchPage.h>

#include <mitkLogMacros.h>
#include <mitkIDataStorageService.h>
#include <QmitkMimeTypes.h>

#include <niftkMultiViewerWidget.h>
#include <niftkDnDDisplayPreferencePage.h>
#include <niftkLoadDataIntoViewerAction.h>
#include <niftkMultiViewerEditor.h>
//#include <niftkSingleViewerEditor.h>

#include <QApplication>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QtPlugin>


namespace niftk
{

PluginActivator* PluginActivator::s_Instance = 0;

//-----------------------------------------------------------------------------
PluginActivator::PluginActivator()
: m_Context(nullptr)
{
  assert(!s_Instance);
  s_Instance = this;
}


//-----------------------------------------------------------------------------
void PluginActivator::start(ctkPluginContext* context)
{
  m_Context = context;
  m_DataStorageServiceTracker = new ctkServiceTracker<mitk::IDataStorageService*>(context);
  m_DataStorageServiceTracker->open();

  BERRY_REGISTER_EXTENSION_CLASS(MultiViewerEditor, context);
//  BERRY_REGISTER_EXTENSION_CLASS(SingleViewerEditor, context);
  BERRY_REGISTER_EXTENSION_CLASS(DnDDisplayPreferencePage, context);
  BERRY_REGISTER_EXTENSION_CLASS(LoadDataIntoViewerAction, context);
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
  m_DataStorageServiceTracker->close();
  delete m_DataStorageServiceTracker;
}


//-----------------------------------------------------------------------------
PluginActivator* PluginActivator::GetInstance()
{
  return s_Instance;
}


//-----------------------------------------------------------------------------
ctkPluginContext* PluginActivator::GetContext()
{
  return m_Context;
}


//-----------------------------------------------------------------------------
mitk::IDataStorageReference::Pointer PluginActivator::GetDataStorageReference() const
{
  mitk::IDataStorageService* dsService = m_DataStorageServiceTracker->getService();

  if (dsService)
  {
    return dsService->GetDataStorage();
  }

  return mitk::IDataStorageReference::Pointer(nullptr);
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer PluginActivator::GetDataStorage() const
{
  mitk::IDataStorageService* dsService = m_DataStorageServiceTracker->getService();

  if (dsService)
  {
    return dsService->GetDataStorage()->GetDataStorage();
  }

  return nullptr;
}

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_dnddisplay, niftk::PluginActivator)
#endif
