/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUSBPortDialog_h
#define niftkUSBPortDialog_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkUSBPortDialog.h>

namespace niftk
{

class NIFTKIGIDATASOURCES_EXPORT USBPortDialog : public IGIInitialisationDialog,
                                                 public Ui_niftkUSBPortDialog

{
  Q_OBJECT

public:

  USBPortDialog(QWidget *parent);
  ~USBPortDialog();

  /**
  * \see IGIInitialisationDialog::GetProperties()
  */
  virtual QMap<QString, QVariant> GetProperties() const override;
};

} // end namespace

#endif
