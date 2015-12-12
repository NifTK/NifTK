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

#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkQtAudioDataDialog.h>

namespace niftk
{

class QtAudioDataDialog : public IGIInitialisationDialog,
                          public Ui_niftkQtAudioDataDialog

{
  Q_OBJECT

public:

  QtAudioDataDialog(QWidget *parent);
  ~QtAudioDataDialog();

private slots:

  void OnCurrentDeviceIndexChanged();
  void OnOKClicked();

private:

  void Update();

};

} // end namespace

#endif
