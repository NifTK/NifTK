/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIPHostPortDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IPHostPortDialog::IPHostPortDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);
  m_HostName->setText("localhost");

  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
IPHostPortDialog::~IPHostPortDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void IPHostPortDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("port", QVariant::fromValue(m_PortNumber->value()));
  props.insert("host", QVariant::fromValue(m_HostName->text()));
  m_Properties = props;
}

} // end namespace
