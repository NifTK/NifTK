/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSingleWidgetEditorPreferencePage.h"
#include <QmitkSingleWidgetEditor.h>

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QCheckBox>
#include <QColorDialog>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

QmitkSingleWidgetEditorPreferencePage::QmitkSingleWidgetEditorPreferencePage()
: m_MainControl(0)
{

}

void QmitkSingleWidgetEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void QmitkSingleWidgetEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_SingleWidgetEditorPreferencesNode = prefService->GetSystemPreferences()->Node(QmitkSingleWidgetEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);
  m_EnableFlexibleZooming = new QCheckBox;
  m_ShowLevelWindowWidget = new QCheckBox;
  m_PACSLikeMouseMode = new QCheckBox;

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow("&Use constrained zooming and padding", m_EnableFlexibleZooming);
  formLayout->addRow("&Show level/window widget", m_ShowLevelWindowWidget);
  formLayout->addRow("&PACS like mouse interactions (select left mouse button action)", m_PACSLikeMouseMode);

  // gradient background
  QLabel* gBName = new QLabel;
  gBName->setText("Gradient background");
  formLayout->addRow(gBName);

  // color
  m_ColorButton1 = new QPushButton;
  m_ColorButton1->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
  m_ColorButton2 = new QPushButton;
  m_ColorButton2->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
  QPushButton* resetButton = new QPushButton;
  resetButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
  resetButton->setText("Reset");

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

  //spacer
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

QWidget* QmitkSingleWidgetEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool QmitkSingleWidgetEditorPreferencePage::PerformOk()
{
  m_SingleWidgetEditorPreferencesNode->Put("first background color style sheet", m_FirstColorStyleSheet.toStdString());
  m_SingleWidgetEditorPreferencesNode->Put("second background color style sheet", m_SecondColorStyleSheet.toStdString());
  m_SingleWidgetEditorPreferencesNode->PutByteArray("first background color", m_FirstColor);
  m_SingleWidgetEditorPreferencesNode->PutByteArray("second background color", m_SecondColor);
  m_SingleWidgetEditorPreferencesNode->PutBool("Use constrained zooming and padding"
                                        , m_EnableFlexibleZooming->isChecked());
  m_SingleWidgetEditorPreferencesNode->PutBool("Show level/window widget", m_ShowLevelWindowWidget->isChecked());
  m_SingleWidgetEditorPreferencesNode->PutBool("PACS like mouse interaction", m_PACSLikeMouseMode->isChecked());

  return true;
}

void QmitkSingleWidgetEditorPreferencePage::PerformCancel()
{

}

void QmitkSingleWidgetEditorPreferencePage::Update()
{
  m_EnableFlexibleZooming->setChecked(m_SingleWidgetEditorPreferencesNode->GetBool("Use constrained zooming and padding", true));
  m_ShowLevelWindowWidget->setChecked(m_SingleWidgetEditorPreferencesNode->GetBool("Show level/window widget", true));
  m_PACSLikeMouseMode->setChecked(m_SingleWidgetEditorPreferencesNode->GetBool("PACS like mouse interaction", false));
  m_FirstColorStyleSheet = QString::fromStdString(m_SingleWidgetEditorPreferencesNode->Get("first background color style sheet", ""));
  m_SecondColorStyleSheet = QString::fromStdString(m_SingleWidgetEditorPreferencesNode->Get("second background color style sheet", ""));
  m_FirstColor = m_SingleWidgetEditorPreferencesNode->GetByteArray("first background color", "");
  m_SecondColor = m_SingleWidgetEditorPreferencesNode->GetByteArray("second background color", "");
  if (m_FirstColorStyleSheet=="")
  {
    m_FirstColorStyleSheet = "background-color:rgb(25,25,25)";
  }
  if (m_SecondColorStyleSheet=="")
  {
    m_SecondColorStyleSheet = "background-color:rgb(127,127,127)";
  }
  if (m_FirstColor=="")
  {
    m_FirstColor = "#191919";
  }
  if (m_SecondColor=="")
  {
    m_SecondColor = "#7F7F7F";
  }
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
}

void QmitkSingleWidgetEditorPreferencePage::FirstColorChanged()
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

void QmitkSingleWidgetEditorPreferencePage::SecondColorChanged()
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

void QmitkSingleWidgetEditorPreferencePage::ResetColors()
{
  m_FirstColorStyleSheet = "background-color:rgb(25,25,25)";
  m_SecondColorStyleSheet = "background-color:rgb(127,127,127)";
  m_FirstColor = "#191919";
  m_SecondColor = "#7F7F7F";
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
}

void QmitkSingleWidgetEditorPreferencePage::UseGradientBackgroundSelected()
{

}

void QmitkSingleWidgetEditorPreferencePage::ColorActionChanged()
{

}

