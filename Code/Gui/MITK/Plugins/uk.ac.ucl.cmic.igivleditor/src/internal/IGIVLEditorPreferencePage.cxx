/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIVLEditorPreferencePage.h"
#include <IGIVLEditor.h>

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QRadioButton>
#include <QColorDialog>
#include <QCheckBox>
#include <ctkPathLineEdit.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>


//-----------------------------------------------------------------------------
const char*           IGIVLEditorPreferencePage::BACKGROUND_COLOR_PREFSKEY  = "background colour";
const unsigned int    IGIVLEditorPreferencePage::DEFAULT_BACKGROUND_COLOR   = 0xFF000000;


//-----------------------------------------------------------------------------
IGIVLEditorPreferencePage::IGIVLEditorPreferencePage()
  : m_MainControl(0)
  , m_BackgroundColourButton(0)
  , m_BackgroundColour(0)
{
}


//-----------------------------------------------------------------------------
void IGIVLEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void IGIVLEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_IGIVLEditorPreferencesNode = prefService->GetSystemPreferences()->Node(IGIVLEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);
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

  QVBoxLayout* vBoxLayout = new QVBoxLayout;
  vBoxLayout->addWidget(colorWidget);
  vBoxLayout->addSpacerItem(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding));

  m_MainControl->setLayout(vBoxLayout);

  bool    ok = false;
  ok = QObject::connect(m_BackgroundColourButton, SIGNAL(clicked()), this, SLOT(OnBackgroundColourClicked()));
  assert(ok);

  this->Update();
}


//-----------------------------------------------------------------------------
void IGIVLEditorPreferencePage::OnBackgroundColourClicked()
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
QWidget* IGIVLEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool IGIVLEditorPreferencePage::PerformOk()
{
  m_IGIVLEditorPreferencesNode->PutInt(BACKGROUND_COLOR_PREFSKEY, m_BackgroundColour);
  return true;
}


//-----------------------------------------------------------------------------
void IGIVLEditorPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void IGIVLEditorPreferencePage::Update()
{
  m_BackgroundColour = m_IGIVLEditorPreferencesNode->GetInt(BACKGROUND_COLOR_PREFSKEY, DEFAULT_BACKGROUND_COLOR);

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
