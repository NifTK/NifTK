/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQtCameraDialog.h"
#include <QCameraInfo>
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
QtCameraDialog::QtCameraDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_FileExtensionComboBox->addItem("JPEG", QVariant::fromValue(QString(".jpg")));
  m_FileExtensionComboBox->addItem("PNG", QVariant::fromValue(QString(".png")));
  m_FileExtensionComboBox->setCurrentIndex(0);

  m_CameraNameComboBox->clear();

  QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
  foreach (const QCameraInfo &cameraInfo, cameras)
  {
    m_CameraNameComboBox->addItem(cameraInfo.description(), QVariant::fromValue(cameraInfo.deviceName()));
  }

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
QtCameraDialog::~QtCameraDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()),
                                this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void QtCameraDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("description", QVariant::fromValue(m_CameraNameComboBox->currentText()));
  props.insert("name", QVariant::fromValue(m_CameraNameComboBox->itemData(m_CameraNameComboBox->currentIndex())));
  props.insert("extension", QVariant::fromValue(m_FileExtensionComboBox->itemData(
                                                m_FileExtensionComboBox->currentIndex())));
  m_Properties = props;
}

} // end namespace
