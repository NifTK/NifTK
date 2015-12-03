/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIPPortDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
IPPortDialog::IPPortDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
IPPortDialog::~IPPortDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void IPPortDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("port", QVariant::fromValue(m_PortNumber->value()));
  m_Properties = props;
}

} // end namespace
