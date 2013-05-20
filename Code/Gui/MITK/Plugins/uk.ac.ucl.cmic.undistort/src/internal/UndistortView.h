/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef UndistortView_h
#define UndistortView_h

#include <berryISelectionListener.h>
#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_UndistortViewControls.h"


class UndistortView : public QmitkBaseView, public Ui::UndistortViewControls
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:  

  UndistortView();
  virtual ~UndistortView();

  static const std::string VIEW_ID;

  virtual void CreateQtPartControl(QWidget *parent);

  // from berry::WorkbenchPart
  virtual void SetFocus();

  /// \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

protected:
  void UpdateNodeTable();


protected slots:


private slots:

  /**
   * \brief We can listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

private:

  /// \brief Retrieves the preferences, and sets the private member variables accordingly.
  void RetrievePreferenceValues();

};

#endif // UndistortView_h
