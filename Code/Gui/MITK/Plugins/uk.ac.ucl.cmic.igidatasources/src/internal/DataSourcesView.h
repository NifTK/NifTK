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

#include <QmitkBaseView.h>
#include <QmitkIGIDataSourceManager.h>
#include <berryIBerryPreferences.h>
#include <ctkDictionary.h>
#include <service/event/ctkEvent.h>


/**
 * \class DataSourcesView
 * \brief User interface to provide Image Guided Surgery functionality.
 * \ingroup uk_ac_ucl_cmic_igidatasources_internal
*/
class DataSourcesView : public QmitkBaseView
{  
  /**
   * this is needed for all Qt objects that should have a Qt meta-object
   * (everything that derives from QObject and wants to have signal/slots)
   */
  Q_OBJECT

public:

  DataSourcesView();
  virtual ~DataSourcesView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igidatasources
   */
  static const std::string VIEW_ID;

  /**
   * \brief Returns the view ID.
   */
  virtual std::string GetViewID() const;

protected:

  /**
   * \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent);

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus();

signals:

  /**
   * \brief We publish an update signal on topic "uk/ac/ucl/cmic/IGIUPDATE" onto the Event Bus so that any other plugin can listen.
   */
  void Updated(const ctkDictionary&);

  /** CTK-bus equivalent of QmitkIGIDataSourceManager's RecordingStarted. Topic is "uk/ac/ucl/cmic/IGIRECORDINGSTARTED". */
  void RecordingStarted(const ctkDictionary&);

protected slots:
  /** Send by footpedal/hotkey plugin. */
  void OnRecordingShouldStart(const ctkEvent& event);
  void OnRecordingShouldStop(const ctkEvent& event);

protected:

private slots:
  
  /**
   * \brief We listen to the QmitkIGIDataSourceManager to publish the update signal.
   */
  void OnUpdateGuiEnd(igtlUint64 timeStamp);

  /** Listens for QmitkIGIDataSourceManager's RecordingStarted signal and forwards it onto the CTK bus. */
  void OnRecordingStarted(QString baseDirectory);

private:

  /**
   * \brief Retrieve's the pref values from preference service, and stored in member variables.
   */
  void RetrievePreferenceValues();

  /**
   * \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
   */
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  QmitkIGIDataSourceManager::Pointer  m_DataSourceManager;
};

#endif // DataSourcesView_h
