/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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

const std::string QmitkMIDASMultiViewEditorPreferencePage::DEFAULT_INTERPOLATION_TYPE("default image interpolation");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_BACKGROUND_COLOUR("midas background colour");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_BACKGROUND_COLOUR_STYLESHEET("midas background colour stylesheet");

const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SLICE_SELECT_TRACKING("midas slice select tracking");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_TIME_SELECT_TRACKING("midas time select tracking");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_MAGNIFICATION_SELECT_TRACKING("midas magnification select tracking");

const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_2D_CURSORS("midas show 2D cursors");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_DIRECTION_ANNOTATIONS("midas show direction annotations");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT("midas show 3D window in multiple window layout");

const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_WINDOW_LAYOUT("midas default window layout");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_REMEMBER_VIEW_SETTINGS_PER_WINDOW_LAYOUT("midas remember view settings of each window layout");

const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_VIEW_ROW_NUMBER("midas default number of view rows");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_DEFAULT_VIEW_COLUMN_NUMBER("midas default number of view columns");

const std::string QmitkMIDASMultiViewEditorPreferencePage::DEFAULT_DROP_TYPE("midas default drop type");

const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_MAGNIFICATION_SLIDER("midas show magnification slider");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_SHOWING_OPTIONS("midas show showing options");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_WINDOW_LAYOUT_CONTROLS("midas show window layout controls");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_VIEW_NUMBER_CONTROLS("midas show view number controls");
const std::string QmitkMIDASMultiViewEditorPreferencePage::MIDAS_SHOW_DROP_TYPE_CONTROLS("midas show drop type widgets");

//-----------------------------------------------------------------------------
QmitkMIDASMultiViewEditorPreferencePage::QmitkMIDASMultiViewEditorPreferencePage()
: m_MainControl(0)
, m_ImageInterpolationComboBox(NULL)
, m_SliceSelectTracking(NULL)
, m_TimeSelectTracking(NULL)
, m_MagnificationSelectTracking(NULL)
, m_Show2DCursorsCheckBox(NULL)
, m_ShowDirectionAnnotationsCheckBox(NULL)
, m_Show3DWindowInMultiWindowLayoutCheckBox(NULL)
, m_DefaultWindowLayoutComboBox(NULL)
, m_RememberEachWindowLayoutsViewSettings(NULL)
, m_DefaultNumberOfViewRowsSpinBox(NULL)
, m_DefaultNumberOfViewColumnsSpinBox(NULL)
, m_DefaultDropType(NULL)
, m_ShowMagnificationSliderCheckBox(NULL)
, m_ShowShowingOptionsCheckBox(NULL)
, m_ShowWindowLayoutControlsCheckBox(NULL)
, m_ShowViewNumberControlsCheckBox(NULL)
, m_ShowDropTypeControlsCheckBox(NULL)
, m_BackgroundColourButton(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkMIDASMultiViewEditorPreferencePage::QmitkMIDASMultiViewEditorPreferencePage(const QmitkMIDASMultiViewEditorPreferencePage& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewEditorPreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewEditorPreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_MIDASMultiViewEditorPreferencesNode = prefService->GetSystemPreferences()->Node(QmitkMIDASMultiViewEditor::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  QFormLayout* formLayout = new QFormLayout;

  m_ImageInterpolationComboBox = new QComboBox(parent);
  m_ImageInterpolationComboBox->insertItem(0, "none");
  m_ImageInterpolationComboBox->insertItem(1, "linear");
  m_ImageInterpolationComboBox->insertItem(2, "cubic");
  formLayout->addRow("image interpolation", m_ImageInterpolationComboBox);

  m_SliceSelectTracking = new QCheckBox(parent);
  formLayout->addRow("slice select tracking", m_SliceSelectTracking);

  m_TimeSelectTracking = new QCheckBox(parent);
  formLayout->addRow("time select tracking", m_TimeSelectTracking);

  m_MagnificationSelectTracking = new QCheckBox(parent);
  formLayout->addRow("magnification select tracking", m_MagnificationSelectTracking);

  m_Show2DCursorsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 2D cursors", m_Show2DCursorsCheckBox);

  m_ShowDirectionAnnotationsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show direction annotations", m_ShowDirectionAnnotationsCheckBox);

  m_Show3DWindowInMultiWindowLayoutCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 3D window in multiple window layout", m_Show3DWindowInMultiWindowLayoutCheckBox);

  m_DefaultWindowLayoutComboBox = new QComboBox(parent);
  m_DefaultWindowLayoutComboBox->insertItem(0, "axial");
  m_DefaultWindowLayoutComboBox->insertItem(1, "sagittal");
  m_DefaultWindowLayoutComboBox->insertItem(2, "coronal");
  m_DefaultWindowLayoutComboBox->insertItem(3, "2x2 orthogonal");
  m_DefaultWindowLayoutComboBox->insertItem(4, "3D");
  m_DefaultWindowLayoutComboBox->insertItem(5, "3 horizontal");
  m_DefaultWindowLayoutComboBox->insertItem(6, "3 vertical");
  m_DefaultWindowLayoutComboBox->insertItem(7, "as acquired (XY plane)");
  formLayout->addRow("default window layout", m_DefaultWindowLayoutComboBox);

  m_RememberEachWindowLayoutsViewSettings = new QCheckBox(parent);
  formLayout->addRow("remember settings of each window layout", m_RememberEachWindowLayoutsViewSettings);

  m_DefaultNumberOfViewRowsSpinBox = new QSpinBox(parent);
  m_DefaultNumberOfViewRowsSpinBox->setMinimum(1);
  m_DefaultNumberOfViewRowsSpinBox->setMaximum(5);
  formLayout->addRow("initial number of view rows", m_DefaultNumberOfViewRowsSpinBox);

  m_DefaultNumberOfViewColumnsSpinBox = new QSpinBox(parent);
  m_DefaultNumberOfViewColumnsSpinBox->setMinimum(1);
  m_DefaultNumberOfViewColumnsSpinBox->setMaximum(5);
  formLayout->addRow("initial number of view columns", m_DefaultNumberOfViewColumnsSpinBox);

  m_DefaultDropType = new QComboBox(parent);
  m_DefaultDropType->insertItem(0, "single");
  m_DefaultDropType->insertItem(1, "multiple");
  m_DefaultDropType->insertItem(2, "all");
  formLayout->addRow("default drop type", m_DefaultDropType);

  m_ShowMagnificationSliderCheckBox = new QCheckBox(parent);
  formLayout->addRow("show magnification slider", m_ShowMagnificationSliderCheckBox);

  m_ShowShowingOptionsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show 'show' options", m_ShowShowingOptionsCheckBox);

  m_ShowWindowLayoutControlsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show window layout controls", m_ShowWindowLayoutControlsCheckBox);

  m_ShowViewNumberControlsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show view number controls", m_ShowViewNumberControlsCheckBox);

  m_ShowDropTypeControlsCheckBox = new QCheckBox(parent);
  formLayout->addRow("show drop type controls", m_ShowDropTypeControlsCheckBox);

  QPushButton* backgroundColourResetButton = new QPushButton(parent);
  backgroundColourResetButton->setText("reset");

  QPushButton* backgroundColorSpecificallyMIDAS = new QPushButton(parent);
  backgroundColorSpecificallyMIDAS->setText("MIDAS default");

  m_BackgroundColourButton = new QPushButton;
  m_BackgroundColourButton->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);

  QGridLayout* backgroundColourWidgetLayout = new QGridLayout;
  backgroundColourWidgetLayout->setContentsMargins(4, 4, 4, 4);
  backgroundColourWidgetLayout->addWidget(m_BackgroundColourButton, 0, 0);
  backgroundColourWidgetLayout->addWidget(backgroundColorSpecificallyMIDAS, 0, 1);
  backgroundColourWidgetLayout->addWidget(backgroundColourResetButton, 0, 2);

  formLayout->addRow("background colour", backgroundColourWidgetLayout);

  m_MainControl->setLayout(formLayout);

  QObject::connect( m_BackgroundColourButton, SIGNAL( clicked() ), this, SLOT( OnBackgroundColourChanged() ) );
  QObject::connect( backgroundColourResetButton, SIGNAL( clicked() ), this, SLOT( OnResetBackgroundColour() ) );
  QObject::connect( backgroundColorSpecificallyMIDAS, SIGNAL( clicked() ), this, SLOT( OnResetMIDASBackgroundColour() ) );

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* QmitkMIDASMultiViewEditorPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewEditorPreferencePage::PerformOk()
{
  m_MIDASMultiViewEditorPreferencesNode->Put(MIDAS_BACKGROUND_COLOUR_STYLESHEET, m_BackgroundColorStyleSheet.toStdString());
  m_MIDASMultiViewEditorPreferencesNode->PutByteArray(MIDAS_BACKGROUND_COLOUR, m_BackgroundColor);
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_VIEW_ROW_NUMBER, m_DefaultNumberOfViewRowsSpinBox->value());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_VIEW_COLUMN_NUMBER, m_DefaultNumberOfViewColumnsSpinBox->value());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(MIDAS_DEFAULT_WINDOW_LAYOUT, m_DefaultWindowLayoutComboBox->currentIndex());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(DEFAULT_INTERPOLATION_TYPE, m_ImageInterpolationComboBox->currentIndex());
  m_MIDASMultiViewEditorPreferencesNode->PutInt(DEFAULT_DROP_TYPE, m_DefaultDropType->currentIndex());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_DROP_TYPE_CONTROLS, m_ShowDropTypeControlsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_SHOWING_OPTIONS, m_ShowShowingOptionsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_WINDOW_LAYOUT_CONTROLS, m_ShowWindowLayoutControlsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_VIEW_NUMBER_CONTROLS, m_ShowViewNumberControlsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_2D_CURSORS, m_Show2DCursorsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_DIRECTION_ANNOTATIONS, m_ShowDirectionAnnotationsCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, m_Show3DWindowInMultiWindowLayoutCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SHOW_MAGNIFICATION_SLIDER, m_ShowMagnificationSliderCheckBox->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_REMEMBER_VIEW_SETTINGS_PER_WINDOW_LAYOUT, m_RememberEachWindowLayoutsViewSettings->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_SLICE_SELECT_TRACKING, m_SliceSelectTracking->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_MAGNIFICATION_SELECT_TRACKING, m_MagnificationSelectTracking->isChecked());
  m_MIDASMultiViewEditorPreferencesNode->PutBool(MIDAS_TIME_SELECT_TRACKING, m_TimeSelectTracking->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewEditorPreferencePage::PerformCancel()
{
}


//-----------------------------------------------------------------------------
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

  m_DefaultNumberOfViewRowsSpinBox->setValue(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_VIEW_ROW_NUMBER, 1));
  m_DefaultNumberOfViewColumnsSpinBox->setValue(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_VIEW_COLUMN_NUMBER, 1));
  m_DefaultWindowLayoutComboBox->setCurrentIndex(m_MIDASMultiViewEditorPreferencesNode->GetInt(MIDAS_DEFAULT_WINDOW_LAYOUT, 2)); // default coronal
  m_ImageInterpolationComboBox->setCurrentIndex(m_MIDASMultiViewEditorPreferencesNode->GetInt(DEFAULT_INTERPOLATION_TYPE, 2));
  m_DefaultDropType->setCurrentIndex(m_MIDASMultiViewEditorPreferencesNode->GetInt(DEFAULT_DROP_TYPE, 0));
  m_ShowDropTypeControlsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_DROP_TYPE_CONTROLS, false));
  m_ShowShowingOptionsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_SHOWING_OPTIONS, true));
  m_ShowWindowLayoutControlsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_WINDOW_LAYOUT_CONTROLS, true));
  m_ShowViewNumberControlsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_VIEW_NUMBER_CONTROLS, true));
  m_Show2DCursorsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_2D_CURSORS, true));
  m_ShowDirectionAnnotationsCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_DIRECTION_ANNOTATIONS, true));
  m_Show3DWindowInMultiWindowLayoutCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
  m_ShowMagnificationSliderCheckBox->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SHOW_MAGNIFICATION_SLIDER, true));
  m_RememberEachWindowLayoutsViewSettings->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_REMEMBER_VIEW_SETTINGS_PER_WINDOW_LAYOUT, true));
  m_SliceSelectTracking->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_SLICE_SELECT_TRACKING, true));
  m_MagnificationSelectTracking->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_MAGNIFICATION_SELECT_TRACKING, true));
  m_TimeSelectTracking->setChecked(m_MIDASMultiViewEditorPreferencesNode->GetBool(MIDAS_TIME_SELECT_TRACKING, true));
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewEditorPreferencePage::OnResetBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(0, 0, 0)";
  m_BackgroundColor = "#000000";
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewEditorPreferencePage::OnResetMIDASBackgroundColour()
{
  m_BackgroundColorStyleSheet = "background-color: rgb(255,250,240)"; // That strange MIDAS off-white colour.
  m_BackgroundColor = "#fffaf0";                                      // That strange MIDAS off-white colour.
  m_BackgroundColourButton->setStyleSheet(m_BackgroundColorStyleSheet);
}
