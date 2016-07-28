/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIVLVideoOverlayEditorPreferencePage.h"
#include "niftkIGIVLVideoOverlayEditor.h"

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QColorDialog>
#include <QCheckBox>
#include <ctkPathLineEdit.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

namespace niftk
{

const QString IGIVLVideoOverlayEditorPreferencePage::CALIBRATION_FILE_NAME("calibration file name");
const QString IGIVLVideoOverlayEditorPreferencePage::BACKGROUND_COLOR_PREFSKEY("background color");
const unsigned int IGIVLVideoOverlayEditorPreferencePage::DEFAULT_BACKGROUND_COLOR(0xFF000000);

//-----------------------------------------------------------------------------
IGIVLVideoOverlayEditorPreferencePage::IGIVLVideoOverlayEditorPreferencePage()
: m_MainControl(nullptr)
, m_CalibrationFileName(nullptr)
, m_BackgroundColourButton(nullptr)
, m_BackgroundColour(0)
{
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_IGIVLVideoOverlayEditorPreferencesNode = prefService->GetSystemPreferences()->Node(IGIVLVideoOverlayEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

  m_CalibrationFileName = new ctkPathLineEdit();
  formLayout->addRow("eye-to-hand matrix file name", m_CalibrationFileName);

  QLabel* colorLabel = new QLabel("Background colour: ");
  colorLabel->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum);
  m_BackgroundColourButton = new QPushButton;
  m_BackgroundColourButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QHBoxLayout* colorWidgetLayout = new QHBoxLayout;
  colorWidgetLayout->setContentsMargins(4,4,4,4);
  colorWidgetLayout->addWidget(colorLabel);
  colorWidgetLayout->addWidget(m_BackgroundColourButton);

  QWidget* colorWidget = new QWidget;
  colorWidget->setLayout(colorWidgetLayout);

  // spacer
  QSpacerItem *spacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  QVBoxLayout* vBoxLayout = new QVBoxLayout;
  vBoxLayout->addLayout(formLayout);
  vBoxLayout->addWidget(colorWidget);
  vBoxLayout->addSpacerItem(spacer);

  m_MainControl->setLayout(vBoxLayout);

  bool ok = false;
  ok = QObject::connect(m_BackgroundColourButton, SIGNAL(clicked()), this, SLOT(OnBackgroundColourClicked()));
  assert(ok);

  this->Update();
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorPreferencePage::OnBackgroundColourClicked()
{
  QColor  color = QColorDialog::getColor();
  m_BackgroundColourButton->setAutoFillBackground(true);

  QString styleSheet = "background-color:rgb(";
  styleSheet.append(QString::number(color.red()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.green()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.blue()));
  styleSheet.append(")");
  m_BackgroundColourButton->setStyleSheet(styleSheet);
  m_BackgroundColour = 0xFF000000 | ((color.blue() & 0xFF) << 16) | ((color.green() & 0xFF) << 8) | (color.red() & 0xFF);
}


//-----------------------------------------------------------------------------
QWidget* IGIVLVideoOverlayEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool IGIVLVideoOverlayEditorPreferencePage::PerformOk()
{
  m_IGIVLVideoOverlayEditorPreferencesNode->Put(CALIBRATION_FILE_NAME, m_CalibrationFileName->currentPath());
  m_IGIVLVideoOverlayEditorPreferencesNode->PutInt(BACKGROUND_COLOR_PREFSKEY, m_BackgroundColour);
  return true;
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayEditorPreferencePage::Update()
{
  m_CalibrationFileName->setCurrentPath(m_IGIVLVideoOverlayEditorPreferencesNode->Get(CALIBRATION_FILE_NAME, ""));
  m_BackgroundColour = m_IGIVLVideoOverlayEditorPreferencesNode->GetInt(BACKGROUND_COLOR_PREFSKEY, DEFAULT_BACKGROUND_COLOR);
  m_BackgroundColourButton->setAutoFillBackground(true);

  QString styleSheet = "background-color:rgb(";
  styleSheet.append(QString::number(m_BackgroundColour & 0x0000FF));
  styleSheet.append(",");
  styleSheet.append(QString::number((m_BackgroundColour & 0x00FF00) >> 8));
  styleSheet.append(",");
  styleSheet.append(QString::number((m_BackgroundColour & 0xFF0000) >> 16));
  styleSheet.append(")");
  m_BackgroundColourButton->setStyleSheet(styleSheet);
}

} // end namespace
