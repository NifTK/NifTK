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
#include <map>
#include <string>
#include <QFuture>
#include <QFutureWatcher>


// forward-decl
namespace niftk
{
class Undistortion;
}


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

  void RunBackgroundProcessing();

signals:
  void SignalDeferredNodeTableUpdate();

private slots:
  void OnDeferredNodeTableUpdate();

  // called by m_NodeTable
  void OnCellDoubleClicked(int row, int column);

  void OnGoButtonClick();

  /**
   * \brief We can listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

  // we connect the future to this slot
  void OnBackgroundProcessFinished();

private:

  void DataStorageEventListener(const mitk::DataNode* node);

  /// \brief Retrieves the preferences, and sets the private member variables accordingly.
  void RetrievePreferenceValues();

  QString                                                m_LastFile;
  std::map<mitk::Image::Pointer, niftk::Undistortion*>   m_UndistortionMap;


  struct WorkItem
  {
    niftk::Undistortion*    m_Proc;
    mitk::Image::Pointer    m_InputImage;
    mitk::Image::Pointer    m_OutputImage;
    std::string             m_OutputNodeName;
    std::string             m_InputNodeName;
  };
  std::vector<WorkItem>                         m_BackgroundQueue;

  QFuture<void>                                 m_BackgroundProcess;
  QFutureWatcher<void>                          m_BackgroundProcessWatcher;
};

#endif // UndistortView_h
