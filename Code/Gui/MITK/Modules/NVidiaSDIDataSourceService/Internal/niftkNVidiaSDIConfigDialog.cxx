/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIConfigDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
NVidiaSDIConfigDialog::NVidiaSDIConfigDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service)
:IGIConfigurationDialog(parent, service)
{
  setupUi(this);

  IGIDataSourceProperties props = m_Service->GetProperties();
  if (props.contains("lag"))
  {
    m_LagSpinBox->setValue(props.value("lag").toInt());
  }

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
NVidiaSDIConfigDialog::~NVidiaSDIConfigDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void NVidiaSDIConfigDialog::OnOKClicked()
{
  QMap<QString, QVariant> props;
  props.insert("lag", QVariant::fromValue(m_LagSpinBox->value()));
  m_Service->SetProperties(props);
}

} // end namespace
