/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIPHostPortDialog_h
#define niftkIPHostPortDialog_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkIPHostPortDialog.h>

namespace niftk
{

class NIFTKIGIDATASOURCES_EXPORT IPHostPortDialog : public IGIInitialisationDialog,
                                                    public Ui_niftkIPHostPortDialog

{
  Q_OBJECT

public:

  IPHostPortDialog(QWidget *parent);
  ~IPHostPortDialog();

private slots:

  void OnOKClicked();

};

} // end namespace

#endif
