/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "uk_ac_ucl_cmic_dnddisplay_Activator.h"

#include <berryPlatform.h>
#include <berryPlatformUI.h>
#include <berryIWorkbenchPage.h>

#include <mitkLogMacros.h>
#include <mitkDataStorage.h>
#include <mitkIDataStorageReference.h>
#include <mitkIDataStorageService.h>
#include <QmitkMimeTypes.h>

#include <niftkMultiViewerWidget.h>
#include <QmitkDnDDisplayPreferencePage.h>
#include <QmitkMultiViewerEditor.h>
//#include <QmitkSingleViewerEditor.h>

#include <QApplication>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QtPlugin>


namespace mitk
{

uk_ac_ucl_cmic_dnddisplay_Activator* uk_ac_ucl_cmic_dnddisplay_Activator::s_Instance = 0;

//-----------------------------------------------------------------------------
uk_ac_ucl_cmic_dnddisplay_Activator::uk_ac_ucl_cmic_dnddisplay_Activator()
: m_PluginContext()
{
  assert(!s_Instance);
  s_Instance = this;
}


//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_dnddisplay_Activator::start(ctkPluginContext* context)
{
  m_PluginContext = context;

  BERRY_REGISTER_EXTENSION_CLASS(QmitkMultiViewerEditor, context);
//  BERRY_REGISTER_EXTENSION_CLASS(QmitkSingleViewerEditor, context);
  BERRY_REGISTER_EXTENSION_CLASS(QmitkDnDDisplayPreferencePage, context);
}


//-----------------------------------------------------------------------------
void uk_ac_ucl_cmic_dnddisplay_Activator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context)
}


//-----------------------------------------------------------------------------
uk_ac_ucl_cmic_dnddisplay_Activator* uk_ac_ucl_cmic_dnddisplay_Activator::GetInstance()
{
  return s_Instance;
}


//-----------------------------------------------------------------------------
ctkPluginContext* uk_ac_ucl_cmic_dnddisplay_Activator::GetPluginContext()
{
  return m_PluginContext;
}

} // end namespace

Q_EXPORT_PLUGIN2(uk_ac_ucl_cmic_dnddisplay, mitk::uk_ac_ucl_cmic_dnddisplay_Activator)
