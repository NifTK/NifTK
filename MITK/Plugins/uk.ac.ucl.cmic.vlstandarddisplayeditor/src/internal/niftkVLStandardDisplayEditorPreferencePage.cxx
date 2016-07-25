/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLStandardDisplayEditorPreferencePage.h"
#include "niftkVLStandardDisplayEditor.h"

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QColorDialog>
#include <QCheckBox>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

namespace niftk
{

const QString VLStandardDisplayEditorPreferencePage::BACKGROUND_COLOR_PREFSKEY("background color");
const unsigned int VLStandardDisplayEditorPreferencePage::DEFAULT_BACKGROUND_COLOR(0xFF000000);

//-----------------------------------------------------------------------------
VLStandardDisplayEditorPreferencePage::VLStandardDisplayEditorPreferencePage()
: m_MainControl(nullptr)
, m_BackgroundColourButton(nullptr)
, m_BackgroundColour(0)
{
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_VLStandardDisplayEditorPreferencesNode = prefService->GetSystemPreferences()->Node(VLStandardDisplayEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);
  QFormLayout *formLayout = new QFormLayout;

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
void VLStandardDisplayEditorPreferencePage::OnBackgroundColourClicked()
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
QWidget* VLStandardDisplayEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool VLStandardDisplayEditorPreferencePage::PerformOk()
{
  m_VLStandardDisplayEditorPreferencesNode->PutInt(BACKGROUND_COLOR_PREFSKEY, m_BackgroundColour);
  return true;
}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditorPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void VLStandardDisplayEditorPreferencePage::Update()
{
  m_BackgroundColour = m_VLStandardDisplayEditorPreferencesNode->GetInt(BACKGROUND_COLOR_PREFSKEY, DEFAULT_BACKGROUND_COLOR);
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
