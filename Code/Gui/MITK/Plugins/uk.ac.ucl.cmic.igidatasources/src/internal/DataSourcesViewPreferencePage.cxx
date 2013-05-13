/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "DataSourcesViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>
#include <QColorDialog>
#include <QSpinBox>
#include <QRadioButton>
#include <QCheckBox>
#include <QDir>
#include <QDesktopServices>
#include <ctkDirectoryButton.h>
#include <QmitkIGIDataSourceManager.h>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string DataSourcesViewPreferencePage::PREFERENCES_NODE_NAME("/uk.ac.ucl.cmic.igidatasources");

//-----------------------------------------------------------------------------
DataSourcesViewPreferencePage::DataSourcesViewPreferencePage()
: m_MainControl(0)
, m_FramesPerSecondSpinBox(0)
, m_ClearDataSpinBox(0)
, m_MilliSecondsTolerance(0)
, m_DirectoryPrefix(0)
, m_SaveOnUpdate(0)
, m_SaveOnReceive(0)
, m_SaveInBackground(0)
, m_Initializing(false)
, m_DataSourcesViewPreferencesNode(0)
{
}


//-----------------------------------------------------------------------------
DataSourcesViewPreferencePage::DataSourcesViewPreferencePage(const DataSourcesViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
DataSourcesViewPreferencePage::~DataSourcesViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
QGridLayout* DataSourcesViewPreferencePage::CreateColourButtonLayout(QPushButton*& button, QPushButton*& resetButton)
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
void DataSourcesViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_DataSourcesViewPreferencesNode = prefService->GetSystemPreferences()->Node(PREFERENCES_NODE_NAME);

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
  m_FramesPerSecondSpinBox->setMaximum(50);

  m_ClearDataSpinBox = new QSpinBox();
  m_ClearDataSpinBox->setMinimum(1);
  m_ClearDataSpinBox->setMaximum(3600);

  m_MilliSecondsTolerance = new QSpinBox();
  m_MilliSecondsTolerance->setMinimum(1);
  m_MilliSecondsTolerance->setMaximum(1000*60*10); // 10 minutes in milliseconds

  m_DirectoryPrefix = new ctkDirectoryButton();

  m_SaveOnUpdate = new QRadioButton();
  m_SaveOnReceive = new QRadioButton();
  m_SaveInBackground = new QCheckBox();

  formLayout->addRow("error colour", errorButtonLayout);
  formLayout->addRow("warning colour", warningButtonLayout);
  formLayout->addRow("OK colour", okButtonLayout);
  formLayout->addRow("refresh rate (per second)", m_FramesPerSecondSpinBox);
  formLayout->addRow("clear data rate (seconds)", m_ClearDataSpinBox);
  formLayout->addRow("timing tolerance (milli-seconds)", m_MilliSecondsTolerance);
  formLayout->addRow("output directory prefix", m_DirectoryPrefix);
  formLayout->addRow("save data each screen update", m_SaveOnUpdate);
  formLayout->addRow("save data as it is received", m_SaveOnReceive);
  formLayout->addRow("save in background", m_SaveInBackground);

  m_MainControl->setLayout(formLayout);
  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* DataSourcesViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool DataSourcesViewPreferencePage::PerformOk()
{
  m_DataSourcesViewPreferencesNode->Put("error colour style sheet", m_ColorStyleSheet[0].toStdString());
  m_DataSourcesViewPreferencesNode->PutByteArray("error colour", m_Color[0]);
  m_DataSourcesViewPreferencesNode->Put("warning colour style sheet", m_ColorStyleSheet[1].toStdString());
  m_DataSourcesViewPreferencesNode->PutByteArray("warning colour", m_Color[1]);
  m_DataSourcesViewPreferencesNode->Put("ok colour style sheet", m_ColorStyleSheet[2].toStdString());
  m_DataSourcesViewPreferencesNode->PutByteArray("ok colour", m_Color[2]);
  m_DataSourcesViewPreferencesNode->PutInt("refresh rate", m_FramesPerSecondSpinBox->value());
  m_DataSourcesViewPreferencesNode->PutInt("clear data rate", m_ClearDataSpinBox->value());
  m_DataSourcesViewPreferencesNode->PutInt("timing tolerance", m_MilliSecondsTolerance->value());
  m_DataSourcesViewPreferencesNode->Put("output directory prefix", m_DirectoryPrefix->directory().toStdString());
  m_DataSourcesViewPreferencesNode->PutBool("save on receive", m_SaveOnReceive->isChecked());
  m_DataSourcesViewPreferencesNode->PutBool("save in background", m_SaveInBackground->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::Update()
{
  m_ColorStyleSheet[0] = QString::fromStdString(m_DataSourcesViewPreferencesNode->Get("error colour style sheet", ""));
  m_Color[0] = m_DataSourcesViewPreferencesNode->GetByteArray("error colour", "");

  if (m_ColorStyleSheet[0]=="" || m_Color[0]=="")
  {
    this->OnResetErrorColour();
  }
  else
  {
    m_ColorPushButton[0]->setStyleSheet(m_ColorStyleSheet[0]);
  }

  m_ColorStyleSheet[1] = QString::fromStdString(m_DataSourcesViewPreferencesNode->Get("warning colour style sheet", ""));
  m_Color[1] = m_DataSourcesViewPreferencesNode->GetByteArray("warning colour", "");

  if (m_ColorStyleSheet[1]=="" || m_Color[1]=="")
  {
    this->OnResetWarningColour();
  }
  else
  {
    m_ColorPushButton[1]->setStyleSheet(m_ColorStyleSheet[1]);
  }

  m_ColorStyleSheet[2] = QString::fromStdString(m_DataSourcesViewPreferencesNode->Get("ok colour style sheet", ""));
  m_Color[2] = m_DataSourcesViewPreferencesNode->GetByteArray("ok colour", "");

  if (m_ColorStyleSheet[2]=="" || m_Color[2]=="")
  {
    this->OnResetOKColour();
  }
  else
  {
    m_ColorPushButton[2]->setStyleSheet(m_ColorStyleSheet[2]);
  }

  m_FramesPerSecondSpinBox->setValue(m_DataSourcesViewPreferencesNode->GetInt("refresh rate", QmitkIGIDataSourceManager::DEFAULT_FRAME_RATE));
  m_ClearDataSpinBox->setValue(m_DataSourcesViewPreferencesNode->GetInt("clear data rate", QmitkIGIDataSourceManager::DEFAULT_CLEAR_RATE));
  m_MilliSecondsTolerance->setValue(m_DataSourcesViewPreferencesNode->GetInt("timing tolerance", QmitkIGIDataSourceManager::DEFAULT_TIMING_TOLERANCE));

  QString path = QString::fromStdString(m_DataSourcesViewPreferencesNode->Get("output directory prefix", ""));

  if (path == "")
  {
    path = QmitkIGIDataSourceManager::GetDefaultPath();
  }
  m_DirectoryPrefix->setDirectory(path);

  m_SaveOnReceive->setChecked(m_DataSourcesViewPreferencesNode->GetBool("save on receive", QmitkIGIDataSourceManager::DEFAULT_SAVE_ON_RECEIPT));
  m_SaveInBackground->setChecked(m_DataSourcesViewPreferencesNode->GetBool("save in background", QmitkIGIDataSourceManager::DEFAULT_SAVE_IN_BACKGROUND));
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnErrorColourChanged()
{
  this->OnColourChanged(0);
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnWarningColourChanged()
{
  this->OnColourChanged(1);
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnOKColourChanged()
{
  this->OnColourChanged(2);
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnColourChanged(int buttonIndex)
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
void DataSourcesViewPreferencePage::OnResetErrorColour()
{
  QColor color = QmitkIGIDataSourceManager::DEFAULT_ERROR_COLOUR;
  this->OnResetColour(0, color);
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnResetWarningColour()
{
  QColor color = QmitkIGIDataSourceManager::DEFAULT_WARNING_COLOUR;
  this->OnResetColour(1, color);
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnResetOKColour()
{
  QColor color = QmitkIGIDataSourceManager::DEFAULT_OK_COLOUR;
  this->OnResetColour(2, color);
}


//-----------------------------------------------------------------------------
void DataSourcesViewPreferencePage::OnResetColour(int buttonIndex, QColor &colorValue)
{
  m_Color[buttonIndex] = (QString("%1").arg(colorValue.name())).toStdString();
  m_ColorStyleSheet[buttonIndex] = QString("background-color: rgb(%1,%2,%3)").arg(colorValue.red()).arg(colorValue.green()).arg(colorValue.blue());
  m_ColorPushButton[buttonIndex]->setStyleSheet(m_ColorStyleSheet[buttonIndex]);
}
