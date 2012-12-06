/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-28 10:00:55 +0100 (Wed, 28 Sep 2011) $
 Revision          : $Revision: 7379 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "SurgicalGuidanceViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>
#include <QColorDialog>
#include <QSpinBox>
#include <QDir>
#include <QDesktopServices>
#include <ctkDirectoryButton.h>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string SurgicalGuidanceViewPreferencePage::PREFERENCES_NODE_NAME("/uk_ac_ucl_cmic_surgicalguidance");

//-----------------------------------------------------------------------------
SurgicalGuidanceViewPreferencePage::SurgicalGuidanceViewPreferencePage()
: m_MainControl(0)
, m_FramesPerSecondSpinBox(0)
, m_DirectoryPrefix(0)
, m_Initializing(false)
, m_SurgicalGuidanceViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
SurgicalGuidanceViewPreferencePage::SurgicalGuidanceViewPreferencePage(const SurgicalGuidanceViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
SurgicalGuidanceViewPreferencePage::~SurgicalGuidanceViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
QGridLayout* SurgicalGuidanceViewPreferencePage::CreateColourButtonLayout(QPushButton*& button, QPushButton*& resetButton)
{
  resetButton = new QPushButton;
  resetButton->setText("reset");

  button = new QPushButton;
  button->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QGridLayout* layout = new QGridLayout;
  layout->setContentsMargins(4,4,4,4);
  layout->addWidget(button, 0, 0);
  layout->addWidget(resetButton, 0, 1);

  return layout;
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_SurgicalGuidanceViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  // Generating error, warning, ok buttons. Colour is set later.

  QPushButton *tmpButton = NULL;
  QPushButton *tmpResetButton = NULL;

  QGridLayout *errorButtonLayout = this->CreateColourButtonLayout(tmpButton, tmpResetButton);
  m_ColorPushButton[0] = tmpButton;
  m_ColorResetPushButton[0] = tmpResetButton;

  QObject::connect( m_ColorPushButton[0], SIGNAL( clicked() ), this, SLOT( OnErrorColourChanged() ) );
  QObject::connect( m_ColorResetPushButton[0], SIGNAL( clicked() ), this, SLOT( OnResetErrorColour() ) );

  QGridLayout *warningButtonLayout = this->CreateColourButtonLayout(tmpButton, tmpResetButton);
  m_ColorPushButton[1] = tmpButton;
  m_ColorResetPushButton[1] = tmpResetButton;

  QObject::connect( m_ColorPushButton[1], SIGNAL( clicked() ), this, SLOT( OnWarningColourChanged() ) );
  QObject::connect( m_ColorResetPushButton[1], SIGNAL( clicked() ), this, SLOT( OnResetWarningColour() ) );

  QGridLayout *okButtonLayout = this->CreateColourButtonLayout(tmpButton, tmpResetButton);
  m_ColorPushButton[2] = tmpButton;
  m_ColorResetPushButton[2] = tmpResetButton;

  QObject::connect( m_ColorPushButton[2], SIGNAL( clicked() ), this, SLOT( OnOKColourChanged() ) );
  QObject::connect( m_ColorResetPushButton[2], SIGNAL( clicked() ), this, SLOT( OnResetOKColour() ) );


  m_FramesPerSecondSpinBox = new QSpinBox();
  m_FramesPerSecondSpinBox->setMinimum(1);
  m_FramesPerSecondSpinBox->setMaximum(25);

  m_DirectoryPrefix = new ctkDirectoryButton();

  formLayout->addRow("error colour", errorButtonLayout);
  formLayout->addRow("warning colour", warningButtonLayout);
  formLayout->addRow("OK colour", okButtonLayout);
  formLayout->addRow("refresh rate (per second)", m_FramesPerSecondSpinBox);
  formLayout->addRow("output directory prefix", m_DirectoryPrefix);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* SurgicalGuidanceViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool SurgicalGuidanceViewPreferencePage::PerformOk()
{
  m_SurgicalGuidanceViewPreferencesNode->Put("error colour style sheet", m_ColorStyleSheet[0].toStdString());
  m_SurgicalGuidanceViewPreferencesNode->PutByteArray("error colour", m_Color[0]);
  m_SurgicalGuidanceViewPreferencesNode->Put("warning colour style sheet", m_ColorStyleSheet[1].toStdString());
  m_SurgicalGuidanceViewPreferencesNode->PutByteArray("warning colour", m_Color[1]);
  m_SurgicalGuidanceViewPreferencesNode->Put("ok colour style sheet", m_ColorStyleSheet[2].toStdString());
  m_SurgicalGuidanceViewPreferencesNode->PutByteArray("ok colour", m_Color[2]);
  m_SurgicalGuidanceViewPreferencesNode->PutInt("refresh rate", m_FramesPerSecondSpinBox->value());
  m_SurgicalGuidanceViewPreferencesNode->Put("output directory prefix", m_DirectoryPrefix->directory().toStdString());
  return true;
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::Update()
{
  m_ColorStyleSheet[0] = QString::fromStdString(m_SurgicalGuidanceViewPreferencesNode->Get("error colour style sheet", ""));
  m_Color[0] = m_SurgicalGuidanceViewPreferencesNode->GetByteArray("error colour", "");

  if (m_ColorStyleSheet[0]=="" || m_Color[0]=="")
  {
    this->OnResetErrorColour();
  }
  else
  {
    m_ColorPushButton[0]->setStyleSheet(m_ColorStyleSheet[0]);
  }

  m_ColorStyleSheet[1] = QString::fromStdString(m_SurgicalGuidanceViewPreferencesNode->Get("warning colour style sheet", ""));
  m_Color[1] = m_SurgicalGuidanceViewPreferencesNode->GetByteArray("warning colour", "");

  if (m_ColorStyleSheet[1]=="" || m_Color[1]=="")
  {
    this->OnResetWarningColour();
  }
  else
  {
    m_ColorPushButton[1]->setStyleSheet(m_ColorStyleSheet[1]);
  }

  m_ColorStyleSheet[2] = QString::fromStdString(m_SurgicalGuidanceViewPreferencesNode->Get("ok colour style sheet", ""));
  m_Color[2] = m_SurgicalGuidanceViewPreferencesNode->GetByteArray("ok colour", "");

  if (m_ColorStyleSheet[2]=="" || m_Color[2]=="")
  {
    this->OnResetOKColour();
  }
  else
  {
    m_ColorPushButton[2]->setStyleSheet(m_ColorStyleSheet[2]);
  }

  m_FramesPerSecondSpinBox->setValue(m_SurgicalGuidanceViewPreferencesNode->GetInt("refresh rate", 2));

  QString path = QString::fromStdString(m_SurgicalGuidanceViewPreferencesNode->Get("output directory prefix", ""));

  if (path == "")
  {
    QDir directory;

    path = QDesktopServices::storageLocation(QDesktopServices::DesktopLocation);
    directory.setPath(path);

    if (!directory.exists())
    {
      path = QDesktopServices::storageLocation(QDesktopServices::DocumentsLocation);
      directory.setPath(path);
    }
    if (!directory.exists())
    {
      path = QDesktopServices::storageLocation(QDesktopServices::HomeLocation);
      directory.setPath(path);
    }
    if (!directory.exists())
    {
      path = QDir::currentPath();
      directory.setPath(path);
    }
    if (!directory.exists())
    {
      path = "";
    }
  }
  m_DirectoryPrefix->setDirectory(path);
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnErrorColourChanged()
{
  this->OnColourChanged(0);
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnWarningColourChanged()
{
  this->OnColourChanged(1);
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnOKColourChanged()
{
  this->OnColourChanged(2);
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnColourChanged(int buttonIndex)
{
  QColor colour = QColorDialog::getColor();
  if (colour.isValid())
  {
    m_ColorPushButton[buttonIndex]->setAutoFillBackground(true);

    QString styleSheet = "background-color: rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");

    m_ColorPushButton[buttonIndex]->setStyleSheet(styleSheet);
    m_ColorStyleSheet[buttonIndex] = styleSheet;

    QStringList defColor;
    defColor << colour.name();

    m_Color[buttonIndex] = defColor.replaceInStrings(";","\\;").join(";").toStdString();
  }
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnResetErrorColour()
{
  this->OnResetColour(0, 255, 0, 0, "ff0000");
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnResetWarningColour()
{
  this->OnResetColour(1, 255, 127, 0, "ff0f00");
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnResetOKColour()
{
  this->OnResetColour(2, 0, 255, 0, "00ff00");
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceViewPreferencePage::OnResetColour(int buttonIndex, unsigned char r, unsigned char g, unsigned char b, std::string hexColour)
{
  m_Color[buttonIndex] = (QString("#%1").arg(QString::fromStdString(hexColour))).toStdString();
  m_ColorStyleSheet[buttonIndex] = QString("background-color: rgb(%1,%2,%3)").arg(r).arg(g).arg(b);
  m_ColorPushButton[buttonIndex]->setStyleSheet(m_ColorStyleSheet[buttonIndex]);
}
