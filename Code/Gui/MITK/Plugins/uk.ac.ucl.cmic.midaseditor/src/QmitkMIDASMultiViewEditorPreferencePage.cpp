/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-02 06:46:00 +0000 (Fri, 02 Dec 2011) $
 Revision          : $Revision: 7905 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASMultiViewEditorPreferencePage.h"
#include "QmitkMIDASMultiViewEditor.h"

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QColorDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QCheckBox>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_NUMBER_ROWS("midas default number of rows");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_NUMBER_COLUMNS("midas default number of columns");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_VIEW("midas default view");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_IMAGE_INTERPOLATION("midas default image interpolation");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_BACKGROUND_COLOUR("midas background colour");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_BACKGROUND_COLOUR_STYLESHEET("midas background colour stylesheet");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_DROP_TYPE("midas default drop type");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_DROP_TYPE_WIDGETS("midas show drop type widgets");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_LAYOUT_BUTTONS("midas show layout buttons");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_3D_VIEW_IN_ORTHOVIEW("midas show 3D view in ortho view");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_2D_CURSORS("midas show 2D cursors");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_MAGNIFICATION_SLIDER("midas show magnification slider");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_REMEMBER_VIEW_SETTINGS_PER_ORIENTATION("midas remember each orientations view settings");

QmitkMIDASMultiViewEditorPreferencePage::QmitkMIDASMultiViewEditorPreferencePage()
: m_MainControl(0)
, m_DefaultNumberOfRowsSpinBox(NULL)
, m_DefaultNumberOfColumnsSpinBox(NULL)
, m_DefaultViewComboBox(NULL)
, m_ImageInterpolationComboBox(NULL)
, m_DefaultDropType(NULL)
, m_ShowDropTypeWidgetsCheckBox(NULL)
, m_ShowLayoutButtonsCheckBox(NULL)
, m_ShowMagnificationSliderCheckBox(NULL)
, m_Show3DInOrthoCheckBox(NULL)
, m_Show2DCursorsCheckBox(NULL)
, m_RememberEachOrientationsViewSettings(NULL)
, m_BackgroundColourButton(NULL)
{

}

QmitkMIDASMultiViewEditorPreferencePage::QmitkMIDASMultiViewEditorPreferencePage(const QmitkMIDASMultiViewEditorPreferencePage& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

void QmitkMIDASMultiViewEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{

}

void QmitkMIDASMultiViewEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_MIDASMultiViewEditorPreferencesNode = prefService->GetSystemPreferences()->Node(QmitkMIDASMultiViewEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout *formLayout = new QFormLayout;

  m_ImageInterpolationComboBox = new QComboBox(parent);
  m_ImageInterpolationComboBox->insertItem(0, "none");
  m_ImageInterpolationComboBox->insertItem(1, "linear");
  m_ImageInterpolationComboBox->insertItem(2, "cubic");
  formLayout->addRow("image interpolation", m_ImageInterpolationComboBox);

  m_DefaultNumberOfRowsSpinBox = new QSpinBox(parent);
  m_DefaultNumberOfRowsSpinBox->setMinimum(1);
  m_DefaultNumberOfRowsSpinBox->setMaximum(5);
  formLayout->addRow("initial number of rows", m_DefaultNumberOfRowsSpinBox);

  m_DefaultNumberOfColumnsSpinBox = new QSpinBox(parent);
  m_DefaultNumberOfColumnsSpinBox->setMinimum(1);
  m_DefaultNumberOfColumnsSpinBox->setMaximum(5);

  formLayout->addRow("initial number of columns", m_DefaultNumberOfColumnsSpinBox);

  m_DefaultViewComboBox = new QComboBox(parent);
  formLayout->addRow("default view", m_DefaultViewComboBox);
  m_DefaultViewComboBox->insertItem(0, "axial");
  m_DefaultViewComboBox->insertItem(1, "sagittal");
  m_DefaultViewComboBox->insertItem(2, "coronal");
  m_DefaultViewComboBox->insertItem(3, "orthogonal");
  // Deliberately getting rid of 3D as a default drop type.
  // If you have this, it's immediately confusing as things like the thumbnail don't work for 3D mappers.
  //m_DefaultViewComboBox->insertItem(4, "3D");
  //m_DefaultViewComboBox->insertItem(5, "as acquired (XY plane)");
  m_DefaultViewComboBox->insertItem(4, "as acquired (XY plane)");

  m_DefaultDropType = new QComboBox(parent);
  formLayout->addRow("default drop type", m_DefaultDropType);
  m_DefaultDropType->insertItem(0, "single");
  m_DefaultDropType->insertItem(1, "multiple");
  m_DefaultDropType->insertItem(2, "all");

  m_ShowDropTypeWidgetsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show drop type check boxes", m_ShowDropTypeWidgetsCheckBox);

  m_ShowLayoutButtonsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show layout buttons", m_ShowLayoutButtonsCheckBox);

  m_ShowMagnificationSliderCheckBox = new QCheckBox(parent);
  formLayout->addRow("show magnification slider", m_ShowMagnificationSliderCheckBox);

  m_Show3DInOrthoCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 3D view in orthoview", m_Show3DInOrthoCheckBox);

  m_Show2DCursorsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 2D cursors", m_Show2DCursorsCheckBox);

  m_RememberEachOrientationsViewSettings = new QCheckBox(parent);
  formLayout->addRow("remember each orientations view settings", m_RememberEachOrientationsViewSettings);

  QPushButton* backgroundColourResetButton = new QPushButton(parent);
  backgroundColourResetButton->setText("reset");

  QPushButton* backgroundColorSpecificallyMIDAS = new QPushButton(parent);
  backgroundColorSpecificallyMIDAS->setText("MIDAS default");

  m_BackgroundColourButton = new QPushButton;
  m_BackgroundColourButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QGridLayout* backgroundColourWidgetLayout = new QGridLayout;
  backgroundColourWidgetLayout->setContentsMargins(4,4,4,4);
  backgroundColourWidgetLayout->addWidget(m_BackgroundColourButton, 0, 0);
  backgroundColourWidgetLayout->addWidget(backgroundColourResetButton, 0, 1);
  backgroundColourWidgetLayout->addWidget(backgroundColorSpecificallyMIDAS, 0, 2);

  formLayout->addRow("background colour", backgroundColourWidgetLayout);

  m_MainControl->setLayout(formLayout);

  QObject::connect( m_BackgroundColourButton, SIGNAL( clicked() ), this, SLOT( OnBackgroundColourChanged() ) );
  QObject::connect( backgroundColourResetButton, SIGNAL( clicked() ), this, SLOT( OnResetBackgroundColour() ) );
  QObject::connect( backgroundColorSpecificallyMIDAS, SIGNAL( clicked() ), this, SLOT( OnResetMIDASBackgroundColour() ) );

  this->Update();
}

QWidget* QmitkMIDASMultiViewEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}

bool QmitkMIDASMultiViewEditorPreferencePage::PerformOk()
{
  m_MIDASMultiViewEditorPreferencesNode->Put(MIDAS_BACKGROUND_COLOUR_STYLESHEET, m_BackgroundColorStyleSheet.toStdString());
  m_MIDASMultiViewEditorPreferencesNode->PutByteArray(MIDAS_BACKGROUND_COLOUR, m_BackgroundColor);
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_NUMBER_ROWS, m_DefaultNumberOfRowsSpinBox->value());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_NUMBER_COLUMNS, m_DefaultNumberOfColumnsSpinBox->value());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_VIEW, m_DefaultViewComboBox->currentIndex());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_IMAGE_INTERPOLATION, m_ImageInterpolationComboBox->currentIndex());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_DROP_TYPE, m_DefaultDropType->currentIndex());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_DROP_TYPE_WIDGETS, m_ShowDropTypeWidgetsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_LAYOUT_BUTTONS, m_ShowLayoutButtonsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_3D_VIEW_IN_ORTHOVIEW, m_Show3DInOrthoCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_2D_CURSORS, m_Show2DCursorsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_MAGNIFICATION_SLIDER, m_ShowMagnificationSliderCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_REMEMBER_VIEW_SETTINGS_PER_ORIENTATION, m_RememberEachOrientationsViewSettings->isChecked());
  return true;
}

void QmitkMIDASMultiViewEditorPreferencePage::PerformCancel()
{

}

void QmitkMIDASMultiViewEditorPreferencePage::Update()
{
  m_BackgroundColorStyleSheet = QString::fromStdString(m_MIDASMultiViewEditorPreferencesNode->Get(MIDAS_BACKGROUND_COLOUR_STYLESHEET, ""));
  m_BackgroundColor = m_MIDASMultiViewEditorPreferencesNode->GetByteArray(MIDAS_BACKGROUND_COLOUR, "");
  if (m_BackgroundColorStyleSheet=="")
  {
    m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  }
  if (m_BackgroundColor=="")
  {
    m_BackgroundColor = "#000000";
  }
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);

  m_DefaultNumberOfRowsSpinBox->setValue(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_NUMBER_ROWS, 1));
  m_DefaultNumberOfColumnsSpinBox->setValue(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_NUMBER_COLUMNS, 1));
  m_DefaultViewComboBox->setCurrentIndex(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_VIEW, 2));
  m_ImageInterpolationComboBox->setCurrentIndex(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_IMAGE_INTERPOLATION, 2));
  m_DefaultDropType->setCurrentIndex(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_DROP_TYPE, 0));
  m_ShowDropTypeWidgetsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_DROP_TYPE_WIDGETS, false));
  m_ShowLayoutButtonsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_LAYOUT_BUTTONS, true));
  m_Show3DInOrthoCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_3D_VIEW_IN_ORTHOVIEW, false));
  m_Show2DCursorsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_2D_CURSORS, true));
  m_ShowMagnificationSliderCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_MAGNIFICATION_SLIDER, true));
  m_RememberEachOrientationsViewSettings->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_REMEMBER_VIEW_SETTINGS_PER_ORIENTATION, true));
}

void QmitkMIDASMultiViewEditorPreferencePage::OnBackgroundColourChanged()
{
  QColor colour = QColorDialog::getColor();
  if (colour.isValid())
  {
    m_BackgroundColourButton->setAutoFillBackground(true);

    QString styleSheet = "background-color: rgb(";
    styleSheet.append(QString::number(colour.red()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.green()));
    styleSheet.append(",");
    styleSheet.append(QString::number(colour.blue()));
    styleSheet.append(")");

    m_BackgroundColourButton->setStyleSheet(styleSheet);
    m_BackgroundColorStyleSheet = styleSheet;

    QStringList backgroundColour;
    backgroundColour << colour.name();

    m_BackgroundColor = backgroundColour.replaceInStrings(";","\\;").join(";").toStdString();
  }
 }

void QmitkMIDASMultiViewEditorPreferencePage::OnResetBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  m_BackgroundColor = "#000000";
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}

void QmitkMIDASMultiViewEditorPreferencePage::OnResetMIDASBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(255,250,240)"; // That strange MIDAS salmon colour.
  m_BackgroundColor = "#fffaf0";                                      // That strange MIDAS salmon colour.
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}
