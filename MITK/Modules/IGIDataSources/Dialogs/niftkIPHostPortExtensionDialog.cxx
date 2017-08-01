/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIPHostPortExtensionDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IPHostPortExtensionDialog::IPHostPortExtensionDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_HostName->setText("localhost");
  m_FileExtensionComboBox->setCurrentIndex(0);

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
IPHostPortExtensionDialog::~IPHostPortExtensionDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void IPHostPortExtensionDialog::AddFileExtension(const QString& name, const QString& extensionWithDot)
{
  m_FileExtensionComboBox->addItem(name, QVariant::fromValue(extensionWithDot));
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
}

} // end namespace
