/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLagDialog_h
#define niftkLagDialog_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIConfigurationDialog.h"
#include <ui_niftkLagDialog.h>

namespace niftk
{

class NIFTKIGIDATASOURCES_EXPORT LagDialog : public IGIConfigurationDialog,
                                             public Ui_niftkLagDialog

{
  Q_OBJECT

public:

  LagDialog(QWidget *parent, niftk::IGIDataSourceI::Pointer service);
  ~LagDialog();

private slots:

  void OnOKClicked();

private:
};

} // end namespace

#endif
