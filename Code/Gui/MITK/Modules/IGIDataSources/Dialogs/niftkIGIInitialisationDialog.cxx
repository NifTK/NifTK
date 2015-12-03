/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIInitialisationDialog.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGIInitialisationDialog::IGIInitialisationDialog(QWidget *parent)
: QDialog(parent)
{

}


//-----------------------------------------------------------------------------
IGIInitialisationDialog::~IGIInitialisationDialog()
{

}


//-----------------------------------------------------------------------------
QMap<QString, QVariant> IGIInitialisationDialog::GetProperties() const
{
  return m_Properties;
}

} // end namespace
