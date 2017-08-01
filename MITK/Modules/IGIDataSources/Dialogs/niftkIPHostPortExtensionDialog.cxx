/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIPHostPortExtensionDialog.h"
#include <QSettings>
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IPHostPortExtensionDialog::IPHostPortExtensionDialog(QWidget *parent,
                                                     const QString& settingsName,
                                                     const QStringList& extensionNames,
                                                     const QStringList& extensionsWithDots
                                                     )
: IGIInitialisationDialog(parent)
, m_SettingsName(settingsName)
{
  setupUi(this);
  assert(extensionNames.size() > 0);
  assert(extensionNames.size() == extensionsWithDots.size());

  for (int i = 0; i < extensionNames.size(); i++)
  {
    m_FileExtensionComboBox->addItem(extensionNames[i], QVariant::fromValue(extensionsWithDots[i]));
  }
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
  m_PortNumber->setValue((settings.value("port", QVariant::fromValue(3200))).toInt());
  settings.endGroup();
}


//-----------------------------------------------------------------------------
IPHostPortExtensionDialog::~IPHostPortExtensionDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void IPHostPortExtensionDialog::SetHostVisible(const bool& isVisible)
{
  this->m_HostName->setVisible(isVisible);
  this->m_HostNameLabel->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void IPHostPortExtensionDialog::SetPortVisible(const bool& isVisible)
{
  this->m_PortNumber->setVisible(isVisible);
  this->m_PortNumberLabel->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void IPHostPortExtensionDialog::SetExtensionVisible(const bool& isVisible)
{
  this->m_FileExtensionComboBox->setVisible(isVisible);
  this->m_FileExtensionLabel->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void IPHostPortExtensionDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("host", QVariant::fromValue(m_HostName->text()));
  props.insert("port", QVariant::fromValue(m_PortNumber->value()));
  props.insert("extension", QVariant::fromValue(m_FileExtensionComboBox->itemData(
                                                m_FileExtensionComboBox->currentIndex())));
  m_Properties = props;

  QSettings settings;
  settings.beginGroup(m_SettingsName);
  settings.setValue("host", m_HostName->text());
  settings.setValue("port", m_PortNumber->value());
  settings.setValue("extension", m_FileExtensionComboBox->itemData(
                                 m_FileExtensionComboBox->currentIndex()));
  settings.endGroup();
}

} // end namespace
