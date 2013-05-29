/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditorPreferencePage.h"
#include <IGIOverlayEditor.h>

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QCheckBox>
#include <QColorDialog>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>


//-----------------------------------------------------------------------------
IGIOverlayEditorPreferencePage::IGIOverlayEditorPreferencePage()
: m_MainControl(0)
, m_ColorButton1(NULL)
, m_ColorButton2(NULL)
{
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_IGIOverlayEditorPreferencesNode = prefService->GetSystemPreferences()->Node(IGIOverlayEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  // gradient background
  QLabel* gBName = new QLabel;
  gBName->setText("gradient background");
  formLayout->addRow(gBName);

  // color
  m_ColorButton1 = new QPushButton;
  m_ColorButton1->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  m_ColorButton2 = new QPushButton;
  m_ColorButton2->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QPushButton* resetButton = new QPushButton;
  resetButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
  resetButton->setText("reset");

  QLabel* colorLabel1 = new QLabel("first color : ");
  colorLabel1->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum);

  QLabel* colorLabel2 = new QLabel("second color: ");
  colorLabel2->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Minimum);

  QHBoxLayout* colorWidgetLayout = new QHBoxLayout;
  colorWidgetLayout->setContentsMargins(4,4,4,4);
  colorWidgetLayout->addWidget(colorLabel1);
  colorWidgetLayout->addWidget(m_ColorButton1);
  colorWidgetLayout->addWidget(colorLabel2);
  colorWidgetLayout->addWidget(m_ColorButton2);
  colorWidgetLayout->addWidget(resetButton);

  QWidget* colorWidget = new QWidget;
  colorWidget->setLayout(colorWidgetLayout);

  // spacer
  QSpacerItem *spacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
  QVBoxLayout* vBoxLayout = new QVBoxLayout;
  vBoxLayout->addLayout(formLayout);
  vBoxLayout->addWidget(colorWidget);
  vBoxLayout->addSpacerItem(spacer);

  m_MainControl->setLayout(vBoxLayout);

  QObject::connect( m_ColorButton1, SIGNAL( clicked() )
    , this, SLOT( FirstColorChanged() ) );

  QObject::connect( m_ColorButton2, SIGNAL( clicked() )
    , this, SLOT( SecondColorChanged() ) );

  QObject::connect( resetButton, SIGNAL( clicked() )
    , this, SLOT( ResetColors() ) );

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* IGIOverlayEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditorPreferencePage::PerformOk()
{
  m_IGIOverlayEditorPreferencesNode->Put("first background color style sheet", m_FirstColorStyleSheet.toStdString());
  m_IGIOverlayEditorPreferencesNode->Put("second background color style sheet", m_SecondColorStyleSheet.toStdString());
  m_IGIOverlayEditorPreferencesNode->PutByteArray("first background color", m_FirstColor);
  m_IGIOverlayEditorPreferencesNode->PutByteArray("second background color", m_SecondColor);
  return true;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::Update()
{
  m_FirstColorStyleSheet = QString::fromStdString(m_IGIOverlayEditorPreferencesNode->Get("first background color style sheet", ""));
  m_SecondColorStyleSheet = QString::fromStdString(m_IGIOverlayEditorPreferencesNode->Get("second background color style sheet", ""));
  m_FirstColor = m_IGIOverlayEditorPreferencesNode->GetByteArray("first background color", "");
  m_SecondColor = m_IGIOverlayEditorPreferencesNode->GetByteArray("second background color", "");
  if (m_FirstColorStyleSheet=="")
  {
    m_FirstColorStyleSheet = "background-color:rgb(0,0,0)";
  }
  if (m_SecondColorStyleSheet=="")
  {
    m_SecondColorStyleSheet = "background-color:rgb(0,0,0)";
  }
  if (m_FirstColor=="")
  {
    m_FirstColor = "#000000";
  }
  if (m_SecondColor=="")
  {
    m_SecondColor = "#000000";
  }
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::FirstColorChanged()
{
  QColor color = QColorDialog::getColor();
  m_ColorButton1->setAutoFillBackground(true);
  QString styleSheet = "background-color:rgb(";

  styleSheet.append(QString::number(color.red()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.green()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.blue()));
  styleSheet.append(")");
  m_ColorButton1->setStyleSheet(styleSheet);

  m_FirstColorStyleSheet = styleSheet;
  QStringList firstColor;
  firstColor << color.name();
  m_FirstColor = firstColor.replaceInStrings(";","\\;").join(";").toStdString();
 }


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::SecondColorChanged()
{
  QColor color = QColorDialog::getColor();
  m_ColorButton2->setAutoFillBackground(true);
  QString styleSheet = "background-color:rgb(";

  styleSheet.append(QString::number(color.red()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.green()));
  styleSheet.append(",");
  styleSheet.append(QString::number(color.blue()));
  styleSheet.append(")");
  m_ColorButton2->setStyleSheet(styleSheet);

  m_SecondColorStyleSheet = styleSheet;
  QStringList secondColor;
  secondColor << color.name();
  m_SecondColor = secondColor.replaceInStrings(";","\\;").join(";").toStdString();
 }


//-----------------------------------------------------------------------------
void IGIOverlayEditorPreferencePage::ResetColors()
{
  m_FirstColorStyleSheet = "background-color:rgb(0,0,0)";
  m_SecondColorStyleSheet = "background-color:rgb(0,0,0)";
  m_FirstColor = "#000000";
  m_SecondColor = "#000000";
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
}

