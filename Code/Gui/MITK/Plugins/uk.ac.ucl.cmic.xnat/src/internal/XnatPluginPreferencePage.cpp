/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-28 10:00:55 +0100 (Wed, 28 Sep 2011) $
 Revision          : $Revision: 7379 $
 Last modified by  : $Author: me $

 Original author   : m.espak@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "XnatPluginPreferencePage.h"

#include <QWidget>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

XnatPluginPreferencePage::XnatPluginPreferencePage()
: m_MainControl(0)
, m_Initializing(false)
{
}

XnatPluginPreferencePage::XnatPluginPreferencePage(const XnatPluginPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

XnatPluginPreferencePage::~XnatPluginPreferencePage()
{

}

void XnatPluginPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void XnatPluginPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService::Pointer prefService 
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_XnatPluginPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.xnat");

  m_MainControl = new QWidget(parent);

  this->Update();

  m_Initializing = false;
}

QWidget* XnatPluginPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool XnatPluginPreferencePage::PerformOk()
{
  return true;
}

void XnatPluginPreferencePage::PerformCancel()
{
}

void XnatPluginPreferencePage::Update()
{
}
