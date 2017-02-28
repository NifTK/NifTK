/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasonixConfigDialog_h
#define niftkUltrasonixConfigDialog_h

#include "niftkIGIConfigurationDialog.h"
#include <ui_niftkUltrasonixConfigDialog.h>

namespace niftk
{

class UltrasonixConfigDialog : public IGIConfigurationDialog,
                               public Ui_niftkUltrasonixConfigDialog

{
  Q_OBJECT

public:

  UltrasonixConfigDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service);
  ~UltrasonixConfigDialog();

private slots:

  void OnOKClicked();

private:
};

} // end namespace

#endif
