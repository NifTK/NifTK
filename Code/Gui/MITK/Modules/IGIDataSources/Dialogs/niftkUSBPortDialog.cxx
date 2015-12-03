/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUSBPortDialog.h"

namespace niftk
{

//-----------------------------------------------------------------------------
USBPortDialog::USBPortDialog(QWidget *parent)
  : IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_PortName->addItem("COM1");
  m_PortName->addItem("COM2");
  m_PortName->addItem("COM3");
  m_PortName->addItem("COM4");
  m_PortName->addItem("COM5");
  m_PortName->addItem("COM6");
  m_PortName->addItem("COM7");
  m_PortName->addItem("COM8");
  m_PortName->addItem("COM9");

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
USBPortDialog::~USBPortDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void USBPortDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("port", QVariant::fromValue(m_PortName->currentText()));
  m_Properties = props;
}

} // end namespace
