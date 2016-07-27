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
#include <mitkDataStorage.h>
#include <mitkIDataStorageReference.h>
#include <mitkIDataStorageService.h>
#include <QmitkMimeTypes.h>

#include <niftkMultiViewerWidget.h>
#include <niftkDnDDisplayPreferencePage.h>
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

  BERRY_REGISTER_EXTENSION_CLASS(MultiViewerEditor, context);
//  BERRY_REGISTER_EXTENSION_CLASS(SingleViewerEditor, context);
  BERRY_REGISTER_EXTENSION_CLASS(DnDDisplayPreferencePage, context);
}


//-----------------------------------------------------------------------------
void PluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
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

}

//-----------------------------------------------------------------------------
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
  Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_dnddisplay, niftk::PluginActivator)
#endif
