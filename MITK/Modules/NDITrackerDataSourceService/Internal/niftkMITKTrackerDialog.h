/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKTrackerDialog_h
#define niftkMITKTrackerDialog_h

#include "niftkIGIInitialisationDialog.h"
#include <ui_niftkMITKTrackerDialog.h>

#include <mitkIPersistenceService.h>

namespace niftk
{

class MITKTrackerDialog : public IGIInitialisationDialog,
                          public Ui_niftkMITKTrackerDialog

{
  Q_OBJECT

public:

  MITKTrackerDialog(QWidget *parent, QString trackerName, int defaultBaudRate);
  ~MITKTrackerDialog();

private slots:

  void OnOKClicked();

private:
  QString m_TrackerName;
  PERSISTENCE_GET_SERVICE_METHOD_MACRO;
};

} // end namespace

#endif
