/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQtCameraDialog_h
#define niftkQtCameraDialog_h

#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkQtCameraDialog.h>

namespace niftk
{

class QtCameraDialog : public IGIInitialisationDialog,
                       public Ui_niftkQtCameraDialog

{
  Q_OBJECT

public:

  QtCameraDialog(QWidget *parent);
  ~QtCameraDialog();

private slots:

  void OnOKClicked();

};

} // end namespace

#endif
