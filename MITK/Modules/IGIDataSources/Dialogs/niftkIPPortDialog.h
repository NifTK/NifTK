/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIPPortDialog_h
#define niftkIPPortDialog_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkIPPortDialog.h>

namespace niftk
{

class NIFTKIGIDATASOURCES_EXPORT IPPortDialog : public IGIInitialisationDialog,
                                                public Ui_niftkIPPortDialog

{
  Q_OBJECT

public:

  IPPortDialog(QWidget *parent);
  ~IPPortDialog();

private slots:

  void OnOKClicked();

};

} // end namespace

#endif
