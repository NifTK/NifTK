/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOpenCVCameraDialog_h
#define niftkOpenCVCameraDialog_h

#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkOpenCVCameraDialog.h>

namespace niftk
{

class OpenCVCameraDialog : public IGIInitialisationDialog,
                           public Ui_niftkOpenCVCameraDialog

{
  Q_OBJECT

public:

  OpenCVCameraDialog(QWidget *parent);
  ~OpenCVCameraDialog();

private slots:

  void OnOKClicked();
};

} // end namespace

#endif
