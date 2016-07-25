/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIVideoOverlayEditorPreferencePage.h"
#include "niftkIGIVideoOverlayEditor.h"

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QRadioButton>
#include <QColorDialog>
#include <QCheckBox>
#include <ctkPathLineEdit.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

namespace niftk
{

const QString IGIVideoOverlayEditorPreferencePage::CALIBRATION_FILE_NAME("calibration file name");
const QString IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_STYLE_SHEET("first background color style sheet");
const QString IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_STYLE_SHEET("second background color style sheet");
const QString IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_COLOUR("first background color");
const QString IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_COLOUR("second background color");

//-----------------------------------------------------------------------------
IGIVideoOverlayEditorPreferencePage::IGIVideoOverlayEditorPreferencePage()
: m_MainControl(nullptr)
, m_CalibrationFileName(nullptr)
, m_ColorButton1(nullptr)
, m_ColorButton2(nullptr)
{
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  m_IGIVideoOverlayEditorPreferencesNode = prefService->GetSystemPreferences()->Node(IGIVideoOverlayEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  m_CalibrationFileName = new ctkPathLineEdit();
  formLayout->addRow("calibration matrix file name", m_CalibrationFileName);

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
QWidget* IGIVideoOverlayEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool IGIVideoOverlayEditorPreferencePage::PerformOk()
{
  m_IGIVideoOverlayEditorPreferencesNode->Put(IGIVideoOverlayEditorPreferencePage::CALIBRATION_FILE_NAME, m_CalibrationFileName->currentPath());
  m_IGIVideoOverlayEditorPreferencesNode->Put(IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_STYLE_SHEET, m_FirstColorStyleSheet);
  m_IGIVideoOverlayEditorPreferencesNode->Put(IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_STYLE_SHEET, m_SecondColorStyleSheet);
  m_IGIVideoOverlayEditorPreferencesNode->Put(IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_COLOUR, m_FirstColor);
  m_IGIVideoOverlayEditorPreferencesNode->Put(IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_COLOUR, m_SecondColor);
  return true;
}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditorPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditorPreferencePage::Update()
{
  m_CalibrationFileName->setCurrentPath(m_IGIVideoOverlayEditorPreferencesNode->Get(IGIVideoOverlayEditorPreferencePage::CALIBRATION_FILE_NAME, ""));
  m_FirstColorStyleSheet = m_IGIVideoOverlayEditorPreferencesNode->Get(IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_STYLE_SHEET, "");
  m_SecondColorStyleSheet = m_IGIVideoOverlayEditorPreferencesNode->Get(IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_STYLE_SHEET, "");
  m_FirstColor = m_IGIVideoOverlayEditorPreferencesNode->Get(IGIVideoOverlayEditorPreferencePage::FIRST_BACKGROUND_COLOUR, "");
  m_SecondColor = m_IGIVideoOverlayEditorPreferencesNode->Get(IGIVideoOverlayEditorPreferencePage::SECOND_BACKGROUND_COLOUR, "");
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
void IGIVideoOverlayEditorPreferencePage::FirstColorChanged()
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
  m_FirstColor = firstColor.replaceInStrings(";","\\;").join(";");
 }


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditorPreferencePage::SecondColorChanged()
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
  m_SecondColor = secondColor.replaceInStrings(";","\\;").join(";");
 }


//-----------------------------------------------------------------------------
void IGIVideoOverlayEditorPreferencePage::ResetColors()
{
  m_FirstColorStyleSheet = "background-color:rgb(0,0,0)";
  m_SecondColorStyleSheet = "background-color:rgb(0,0,0)";
  m_FirstColor = "#000000";
  m_SecondColor = "#000000";
  m_ColorButton1->setStyleSheet(m_FirstColorStyleSheet);
  m_ColorButton2->setStyleSheet(m_SecondColorStyleSheet);
}

} // end namespace
