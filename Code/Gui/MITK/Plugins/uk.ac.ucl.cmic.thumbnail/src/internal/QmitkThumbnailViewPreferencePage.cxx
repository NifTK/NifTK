/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkThumbnailViewPreferencePage.h"

#include <QFormLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>

#include <berryIPreferencesService.h>
#include <berryPlatform.h>

const QString QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_THICKNESS("thumbnail view box thickness");
const QString QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_OPACITY("thumbnail view box opacity");
const QString QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_LAYER("thumbnail view box layer");
const QString QmitkThumbnailViewPreferencePage::THUMBNAIL_TRACK_ONLY_MAIN_WINDOWS("thumbnail track only main windows");

//-----------------------------------------------------------------------------
QmitkThumbnailViewPreferencePage::QmitkThumbnailViewPreferencePage()
: m_MainControl(0)
, m_BoxThickness(0)
, m_BoxOpacity(0)
, m_BoxLayer(0)
, m_TrackOnlyMainWindows(0)
, m_Initializing(false)
{

}


//-----------------------------------------------------------------------------
QmitkThumbnailViewPreferencePage::QmitkThumbnailViewPreferencePage(const QmitkThumbnailViewPreferencePage& other)
: berry::Object(), QObject()
{
  Q_UNUSED(other)
  throw std::runtime_error("QmitkThumbnailViewPreferencePage copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkThumbnailViewPreferencePage::~QmitkThumbnailViewPreferencePage()
{

}


//-----------------------------------------------------------------------------
void QmitkThumbnailViewPreferencePage::Init(berry::IWorkbench::Pointer )
{

}


//-----------------------------------------------------------------------------
void QmitkThumbnailViewPreferencePage::CreateQtControl(QWidget* parent)
{
  m_Initializing = true;

  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  m_ThumbnailPreferencesNode = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.thumbnail");

  m_MainControl = new QWidget(parent);

  m_BoxThickness = new QSpinBox();
  m_BoxThickness->setMinimum(1);
  m_BoxThickness->setMaximum(50);
  m_BoxThickness->setSingleStep(1);
  m_BoxThickness->setValue(1);

  m_BoxOpacity = new QDoubleSpinBox();
  m_BoxOpacity->setMinimum(0);
  m_BoxOpacity->setMaximum(1);
  m_BoxOpacity->setSingleStep(0.1);
  m_BoxOpacity->setValue(1);

  m_BoxLayer = new QSpinBox();
  m_BoxLayer->setMinimum(0);
  m_BoxLayer->setMaximum(1000);
  m_BoxLayer->setSingleStep(1);
  m_BoxLayer->setValue(99);

  m_TrackOnlyMainWindows = new QCheckBox();
  m_TrackOnlyMainWindows->setChecked(true);
  QString trackOnlyMainWindowsToolTip =
      "If checked, the thumbnail viewer will not track windows\n"
      "that are not on the main display, but on some side view.";
  m_TrackOnlyMainWindows->setToolTip(trackOnlyMainWindowsToolTip);

  QFormLayout *formLayout = new QFormLayout;
  formLayout->addRow("line width", m_BoxThickness );
  formLayout->addRow("line opacity", m_BoxOpacity );

  formLayout->addRow("rendering layer", m_BoxLayer );
  formLayout->addRow("track only windows of the main display", m_TrackOnlyMainWindows);
  formLayout->labelForField(m_TrackOnlyMainWindows)->setToolTip(trackOnlyMainWindowsToolTip);

  m_MainControl->setLayout(formLayout);

  this->Update();

  m_Initializing = false;
}


//-----------------------------------------------------------------------------
QWidget* QmitkThumbnailViewPreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool QmitkThumbnailViewPreferencePage::PerformOk()
{
  m_ThumbnailPreferencesNode->PutDouble(THUMBNAIL_BOX_OPACITY, m_BoxOpacity->value());
  m_ThumbnailPreferencesNode->PutInt(THUMBNAIL_BOX_THICKNESS, m_BoxThickness->value());
  m_ThumbnailPreferencesNode->PutInt(THUMBNAIL_BOX_LAYER, m_BoxLayer->value());
  m_ThumbnailPreferencesNode->PutBool(THUMBNAIL_TRACK_ONLY_MAIN_WINDOWS, m_TrackOnlyMainWindows->isChecked());
  return true;
}


//-----------------------------------------------------------------------------
void QmitkThumbnailViewPreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void QmitkThumbnailViewPreferencePage::Update()
{
  m_BoxThickness->setValue(m_ThumbnailPreferencesNode->GetInt(THUMBNAIL_BOX_THICKNESS, 1));
  m_BoxLayer->setValue(m_ThumbnailPreferencesNode->GetInt(THUMBNAIL_BOX_LAYER, 99));
  m_BoxOpacity->setValue(m_ThumbnailPreferencesNode->GetDouble(THUMBNAIL_BOX_OPACITY, 1));
  m_TrackOnlyMainWindows->setChecked(m_ThumbnailPreferencesNode->GetBool(THUMBNAIL_TRACK_ONLY_MAIN_WINDOWS, true));
}
