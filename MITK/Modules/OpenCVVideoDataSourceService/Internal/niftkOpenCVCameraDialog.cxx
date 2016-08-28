/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVCameraDialog.h"
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVCameraDialog::OpenCVCameraDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_FileExtensionComboBox->addItem("JPEG", QVariant::fromValue(QString(".jpg")));
  m_FileExtensionComboBox->addItem("PNG", QVariant::fromValue(QString(".png")));
  m_FileExtensionComboBox->setCurrentIndex(0);

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
OpenCVCameraDialog::~OpenCVCameraDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void OpenCVCameraDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("extension", QVariant::fromValue(m_FileExtensionComboBox->itemData(
                                                  m_FileExtensionComboBox->currentIndex())));
  m_Properties = props;
}

} // end namespace
