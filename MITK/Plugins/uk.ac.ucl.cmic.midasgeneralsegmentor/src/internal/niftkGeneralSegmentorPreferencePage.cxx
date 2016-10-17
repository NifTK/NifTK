/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorPreferencePage.h"
#include "niftkGeneralSegmentorView.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QMessageBox>
#include <QPushButton>
#include <QColorDialog>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <niftkBaseSegmentorView.h>

namespace niftk
{

//-----------------------------------------------------------------------------
GeneralSegmentorPreferencePage::GeneralSegmentorPreferencePage()
: m_MainControl(0)
, m_Initializing(false)
{

}


//-----------------------------------------------------------------------------
GeneralSegmentorPreferencePage::GeneralSegmentorPreferencePage(const GeneralSegmentorPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
GeneralSegmentorPreferencePage::~GeneralSegmentorPreferencePage()
{

}


//-----------------------------------------------------------------------------
void GeneralSegmentorPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void GeneralSegmentorPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  m_GeneralSegmentorPreferencesNode = prefService->GetSystemPreferences()->Node(niftk::GeneralSegmentorView::VIEW_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  QPushButton* defaultColorResetButton = new QPushButton;
  defaultColorResetButton->setText("reset");

  m_DefaultColorPushButton = new QPushButton;
  m_DefaultColorPushButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QGridLayout* defaultColorLayout = new QGridLayout;
  defaultColorLayout->setContentsMargins(4,4,4,4);
  defaultColorLayout->addWidget(m_DefaultColorPushButton, 0, 0);
  defaultColorLayout->addWidget(defaultColorResetButton, 0, 1);

  formLayout->addRow("default outline colour", defaultColorLayout);

  m_MainControl->setLayout(formLayout);

  QObject::connect( m_DefaultColorPushButton, SIGNAL( clicked() ), this, SLOT( OnDefaultColourChanged() ) );
  QObject::connect( defaultColorResetButton, SIGNAL( clicked() ), this, SLOT( OnResetDefaultColour() ) );

  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* GeneralSegmentorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool GeneralSegmentorPreferencePage::PerformOk()
{
  m_GeneralSegmentorPreferencesNode->Put(BaseSegmentorView::DEFAULT_COLOUR_STYLE_SHEET, m_DefauleColorStyleSheet);
  m_GeneralSegmentorPreferencesNode->Put(BaseSegmentorView::DEFAULT_COLOUR, m_DefaultColor);

  return true;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void GeneralSegmentorPreferencePage::Update()
{
  m_DefauleColorStyleSheet = m_GeneralSegmentorPreferencesNode->Get(BaseSegmentorView::DEFAULT_COLOUR_STYLE_SHEET, "");
  m_DefaultColor = m_GeneralSegmentorPreferencesNode->Get(BaseSegmentorView::DEFAULT_COLOUR, "");
  if (m_DefauleColorStyleSheet=="")
  {
    m_DefauleColorStyleSheet = "background-color: rgb(0,255,0)";
  }
  if (m_DefaultColor=="")
  {
    m_DefaultColor = "#00ff00";
  }
  m_DefaultColorPushButton->setStyleSheet(m_DefauleColorStyleSheet);
}


//-----------------------------------------------------------------------------
void GeneralSegmentorPreferencePage::OnDefaultColourChanged()
{
  QColor colour = QColorDialog::getColor();
  if (colour.isValid())
  {
    m_DefaultColorPushButton->setAutoFillBackground(true);

    QString styleSheet = "background-color: rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");

    m_DefaultColorPushButton->setStyleSheet(styleSheet);
    m_DefauleColorStyleSheet = styleSheet;

    QStringList defColor;
    defColor << colour.name();

    m_DefaultColor = defColor.replaceInStrings(";","\\;").join(";");
  }
}


//-----------------------------------------------------------------------------
void GeneralSegmentorPreferencePage::OnResetDefaultColour()
{
  m_DefauleColorStyleSheet = "background-color: rgb(0,255,0)";
  m_DefaultColor = "#00ff00";
  m_DefaultColorPushButton->setStyleSheet(m_DefauleColorStyleSheet);
}

}
