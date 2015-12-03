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
}


//-----------------------------------------------------------------------------
USBPortDialog::~USBPortDialog()
{

}


//-----------------------------------------------------------------------------
QMap<QString, QVariant> USBPortDialog::GetProperties() const
{
  QMap<QString, QVariant> empty;
  return empty;
}

} // end namespace
