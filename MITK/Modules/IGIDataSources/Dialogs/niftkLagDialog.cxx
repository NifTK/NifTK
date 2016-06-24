/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLagDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
LagDialog::LagDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service)
:IGIConfigurationDialog(parent, service)
{
  setupUi(this);

  IGIDataSourceProperties props = m_Service->GetProperties();
  if (props.contains("lag"))
  {
    m_LagSpinBox->setValue(props.value("lag").toInt());
  }
  bool ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
LagDialog::~LagDialog()
{
  bool ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void LagDialog::OnOKClicked()
{
  QMap<QString, QVariant> props;
  props.insert("lag", QVariant::fromValue(m_LagSpinBox->value()));
  m_Service->SetProperties(props);
}

} // end namespace
