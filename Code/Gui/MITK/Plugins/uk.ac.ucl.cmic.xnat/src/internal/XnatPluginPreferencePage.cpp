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
#include <QDebug>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_NAME("download directory");
const std::string XnatPluginPreferencePage::DOWNLOAD_DIRECTORY_DEFAULT(".");

XnatPluginPreferencePage::XnatPluginPreferencePage()
: m_Initializing(false)
, m_MainControl(0)
, m_Controls(0)
{
}

XnatPluginPreferencePage::~XnatPluginPreferencePage()
{
  if (m_Controls)
  {
    delete m_MainControl;
    delete m_Controls;
  }
}

void XnatPluginPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void XnatPluginPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;
  berry::IPreferencesService::Pointer prefService =
      berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_XnatPluginPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.xnat");

  if (!m_Controls)
  {
    m_MainControl = new QWidget(parent);

    // Create UI
    m_Controls = new Ui::XnatPluginPreferencePage();
    m_Controls->setupUi(m_MainControl);
  }

  this->Update();

  m_Initializing = false;
}

QWidget* XnatPluginPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool XnatPluginPreferencePage::PerformOk()
{
  QString downloadDirectory = m_Controls->dirBtnDownloadDirectory->directory();

  m_XnatPluginPreferencesNode->Put(DOWNLOAD_DIRECTORY_NAME, downloadDirectory.toStdString());

  return true;
}

void XnatPluginPreferencePage::PerformCancel()
{
}

void XnatPluginPreferencePage::Update()
{
  std::string downloadDirectory = m_XnatPluginPreferencesNode->Get(DOWNLOAD_DIRECTORY_NAME, DOWNLOAD_DIRECTORY_DEFAULT);
  m_Controls->dirBtnDownloadDirectory->setDirectory(QString::fromStdString(downloadDirectory));
}
