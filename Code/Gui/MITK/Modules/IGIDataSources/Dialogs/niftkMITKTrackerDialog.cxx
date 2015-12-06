/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDialog.h"

namespace niftk
{

//-----------------------------------------------------------------------------
MITKTrackerDialog::MITKTrackerDialog(QWidget *parent)
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
  m_PortName->addItem("COM10");
  m_PortName->addItem("COM11");
  m_PortName->addItem("COM12");
  m_PortName->addItem("COM13");

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
MITKTrackerDialog::~MITKTrackerDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void MITKTrackerDialog::OnOKClicked()
{
  int currentSelection = m_PortName->currentIndex() + 1;

  IGIDataSourceProperties props;
  props.insert("port", QVariant::fromValue(currentSelection));
  props.insert("file", QVariant::fromValue(m_FileOpen->currentPath()));
  m_Properties = props;
}

} // end namespace
