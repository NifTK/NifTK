/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNVidiaSDIConfigDialog_h
#define niftkNVidiaSDIConfigDialog_h

#include "niftkIGIConfigurationDialog.h"
#include <ui_niftkNVidiaSDIConfigDialog.h>

namespace niftk
{

class NVidiaSDIConfigDialog : public IGIConfigurationDialog,
                              public Ui_niftkNVidiaSDIConfigDialog

{
  Q_OBJECT

public:

  NVidiaSDIConfigDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service);
  ~NVidiaSDIConfigDialog();

private slots:

  void OnOKClicked();

private:
};

} // end namespace

#endif
