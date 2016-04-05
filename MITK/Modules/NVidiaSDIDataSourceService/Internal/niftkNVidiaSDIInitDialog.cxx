/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIInitDialog.h"

#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
NVidiaSDIInitDialog::NVidiaSDIInitDialog(QWidget *parent)
:IGIInitialisationDialog(parent)
{
  setupUi(this);

  bool ok = false;
  ok = QObject::connect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
NVidiaSDIInitDialog::~NVidiaSDIInitDialog()
{
  bool ok = false;
  ok = QObject::disconnect(m_DialogButtons, SIGNAL(accepted()), this, SLOT(OnOKClicked()));
  assert(ok);
}


//-----------------------------------------------------------------------------
void NVidiaSDIInitDialog::OnOKClicked()
{
  IGIDataSourceProperties props;
  props.insert("mode", QVariant::fromValue(FieldModeComboBox->currentIndex()));
  m_Properties = props;
}

} // end namespace
