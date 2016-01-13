/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNVidiaSDIInitDialog_h
#define niftkNVidiaSDIInitDialog_h

#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkNVidiaSDIInitDialog.h>

namespace niftk
{

class NVidiaSDIInitDialog : public IGIInitialisationDialog,
                            public Ui_niftkNVidiaSDIInitDialog

{
  Q_OBJECT

public:

  NVidiaSDIInitDialog(QWidget *parent);
  ~NVidiaSDIInitDialog();

private slots:

  void OnOKClicked();

};

} // end namespace

#endif
