/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef DataSourcesView_h
#define DataSourcesView_h

#include <niftkBaseView.h>
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataSourceManagerWidget.h>
#include <berryIBerryPreferences.h>
#include <ctkDictionary.h>
#include <service/event/ctkEvent.h>

namespace niftk
{

/**
* \class DataSourcesView
* \brief User interface to provide Image Guided Surgery functionality.
* \ingroup uk_ac_ucl_cmic_igidatasources_internal
*/
class DataSourcesView : public niftk::BaseView
{  
  /**
  * this is needed for all Qt objects that should have a Qt meta-object
  * (everything that derives from QObject and wants to have signal/slots)
  */
  Q_OBJECT

public:

  /**
   * \brief Each View for a plugin has its own globally unique ID, this one is
   * "uk.ac.ucl.cmic.igidatasources" and the .cxx file and plugin.xml should match.
   */
  static const QString VIEW_ID;

  DataSourcesView();
  virtual ~DataSourcesView();

protected:

  /**
  * \brief Called by framework, this method creates all the controls for this view
  */
  virtual void CreateQtPartControl(QWidget *parent) override;

  /**
  * \brief Called by framework, sets the focus on a specific widget.
  */
  virtual void SetFocus() override;

signals:

  /**
  * \brief We publish an update signal on topic "uk/ac/ucl/cmic/IGIUPDATE" onto the Event Bus so that any other plugin can listen.
  */
  void Updated(const ctkDictionary&);

  /**
  * \brief CTK-bus equivalent of IGIDataSourceManager's RecordingStarted. Topic is "uk/ac/ucl/cmic/IGIRECORDINGSTARTED".
  */
  void RecordingStarted(const ctkDictionary&);

  /**
  * \brief CTK-bus equivalent of IGIDataSourceManager's RecordingStopped. Topic is "uk/ac/ucl/cmic/IGIRECORDINGSTOPPED".
  */
  void RecordingStopped(const ctkDictionary&);

protected slots:

  /** Sent by other plugins, requesting a pause/restart. */
  void OnUpdateShouldPause(const ctkEvent& event);
  void OnUpdateShouldRestart(const ctkEvent& event);

  /** Sent by footswitch. */
  void OnToggleRecording(const ctkEvent& event);

protected:

private slots:
  
  /**
  * \brief We listen to the IGIDataSourceManager to publish the update signal.
  */
  void OnUpdateGuiEnd(niftk::IGIDataType::IGITimeType timeStamp);

  /**
  * \brief Listens for IGIDataSourceManager's RecordingStarted signal and forwards it onto the CTK bus.
  */
  void OnRecordingStarted(QString baseDirectory);

  /**
  * \brief Listens for IGIDataSourceManager's RecordingStopped signal and forwards it onto the CTK bus.
  */
  void OnRecordingStopped();

private:

  /**
  * \brief Retrieve's the pref values from preference service, and store in member variables.
  */
  void RetrievePreferenceValues();

  /**
  * \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
  */
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*) override;

  niftk::IGIDataSourceManagerWidget* m_DataSourceManagerWidget;
  bool                               m_SetupWasCalled;
};

} // end namespace

#endif // DataSourcesView_h
