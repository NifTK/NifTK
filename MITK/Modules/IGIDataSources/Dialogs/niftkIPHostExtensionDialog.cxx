/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIPHostExtensionDialog.h"
#include <QSettings>
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IPHostExtensionDialog::IPHostExtensionDialog(QWidget *parent, const QString& settingsName)
:IGIInitialisationDialog(parent)
, m_SettingsName(settingsName)
{
  setupUi(this);
  m_FileExtensionComboBox->addItem("JPEG", QVariant::fromValue(QString(".jpg")));
  m_FileExtensionComboBox->addItem("PNG", QVariant::fromValue(QString(".png")));
  m_FileExtensionComboBox->setCurrentIndex(0);

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);

  QSettings settings;
  settings.beginGroup(m_SettingsName);
  m_HostName->setText(settings.value("host", "localhost").toString());

  int position = m_FileExtensionComboBox->findData(QVariant(settings.value("extension", ".jpg")));
  if (position != -1)
  {
    m_FileExtensionComboBox->setCurrentIndex(position);
  }

  settings.endGroup();
}


//-----------------------------------------------------------------------------
IPHostExtensionDialog::~IPHostExtensionDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void IPHostExtensionDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("host", QVariant::fromValue(m_HostName->text()));
  props.insert("extension", QVariant::fromValue(m_FileExtensionComboBox->itemData(
                                                m_FileExtensionComboBox->currentIndex())));
  m_Properties = props;

  QSettings settings;
  settings.beginGroup(m_SettingsName);
  settings.setValue("host", m_HostName->text());
  settings.setValue("extension", m_FileExtensionComboBox->itemData(m_FileExtensionComboBox->currentIndex()));
  settings.endGroup();
}

} // end namespace
