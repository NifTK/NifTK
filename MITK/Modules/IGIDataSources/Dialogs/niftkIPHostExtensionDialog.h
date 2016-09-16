/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIPHostExtensionDialog_h
#define niftkIPHostExtensionDialog_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkIPHostExtensionDialog.h>

namespace niftk
{

class NIFTKIGIDATASOURCES_EXPORT IPHostExtensionDialog : public IGIInitialisationDialog,
                                                         public Ui_niftkIPHostExtensionDialog

{
  Q_OBJECT

public:

  IPHostExtensionDialog(QWidget *parent);
  ~IPHostExtensionDialog();

private slots:

  void OnOKClicked();

};

} // end namespace

#endif
