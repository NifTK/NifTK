/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

// Qmitk
#include "SurgicalGuidanceView.h"

const std::string SurgicalGuidanceView::VIEW_ID = "uk.ac.ucl.cmic.surgicalguidance";

//-----------------------------------------------------------------------------
SurgicalGuidanceView::SurgicalGuidanceView()
{
}


//-----------------------------------------------------------------------------
SurgicalGuidanceView::~SurgicalGuidanceView()
{
}


//-----------------------------------------------------------------------------
std::string SurgicalGuidanceView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceView::CreateQtPartControl( QWidget *parent )
{
  m_DataSourceManager = QmitkIGIDataSourceManager::New();
  m_DataSourceManager->setupUi(parent);
  m_DataSourceManager->SetStdMultiWidget(this->GetActiveStdMultiWidget());
  m_DataSourceManager->SetDataStorage(this->GetDataStorage());

  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceView::SetFocus()
{
  m_DataSourceManager->setFocus();
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    QString path = QString::fromStdString(prefs->Get("output directory prefix", ""));
    if (path == "")
    {
      path = QmitkIGIDataSourceManager::GetDefaultPath();
    }
    QColor errorColour = QmitkIGIDataSourceManager::DEFAULT_ERROR_COLOUR;
    std::string errorColourName = prefs->Get("error colour", "");
    if (errorColourName != "")
    {
      errorColour.setNamedColor(QString::fromStdString(errorColourName));
    }
    QColor warningColour = QmitkIGIDataSourceManager::DEFAULT_WARNING_COLOUR;
    std::string warningColourName = prefs->Get("warning colour", "");
    if (warningColourName != "")
    {
      warningColour.setNamedColor(QString::fromStdString(warningColourName));
    }
    QColor okColour = QmitkIGIDataSourceManager::DEFAULT_OK_COLOUR;
    std::string okColourName = prefs->Get("ok colour", "");
    if (okColourName != "")
    {
      okColour.setNamedColor(QString::fromStdString(okColourName));
    }

    int refreshRate = prefs->GetInt("refresh rate", QmitkIGIDataSourceManager::DEFAULT_FRAME_RATE);
    int clearRate = prefs->GetInt("clear data rate", QmitkIGIDataSourceManager::DEFAULT_CLEAR_RATE);
    bool saveOnReceipt = prefs->GetBool("save on receive", QmitkIGIDataSourceManager::DEFAULT_SAVE_ON_RECEIPT);
    bool saveInBackground = prefs->GetBool("save in background", QmitkIGIDataSourceManager::DEFAULT_SAVE_IN_BACKGROUND);

    m_DataSourceManager->SetDirectoryPrefix(path);
    m_DataSourceManager->SetFramesPerSecond(refreshRate);
    m_DataSourceManager->SetErrorColour(errorColour);
    m_DataSourceManager->SetWarningColour(warningColour);
    m_DataSourceManager->SetOKColour(okColour);
    m_DataSourceManager->SetClearDataRate(clearRate);
    m_DataSourceManager->SetSaveOnReceipt(saveOnReceipt);
    m_DataSourceManager->SetSaveInBackground(saveInBackground);
  }
  else
  {
    QString defaultPath = QmitkIGIDataSourceManager::GetDefaultPath();
    QColor defaultErrorColor = QmitkIGIDataSourceManager::DEFAULT_ERROR_COLOUR;
    QColor defaultWarningColor = QmitkIGIDataSourceManager::DEFAULT_WARNING_COLOUR;
    QColor defaultOKColor = QmitkIGIDataSourceManager::DEFAULT_OK_COLOUR;

    m_DataSourceManager->SetDirectoryPrefix(defaultPath);
    m_DataSourceManager->SetFramesPerSecond(QmitkIGIDataSourceManager::DEFAULT_FRAME_RATE);
    m_DataSourceManager->SetErrorColour(defaultErrorColor);
    m_DataSourceManager->SetWarningColour(defaultWarningColor);
    m_DataSourceManager->SetOKColour(defaultOKColor);
    m_DataSourceManager->SetClearDataRate(QmitkIGIDataSourceManager::DEFAULT_CLEAR_RATE);
    m_DataSourceManager->SetSaveOnReceipt(QmitkIGIDataSourceManager::DEFAULT_SAVE_ON_RECEIPT);
    m_DataSourceManager->SetSaveInBackground(QmitkIGIDataSourceManager::DEFAULT_SAVE_IN_BACKGROUND);
  }
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}

