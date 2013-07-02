/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef SurfaceReconView_h
#define SurfaceReconView_h

#include <QmitkBaseView.h>
#include <SurfaceReconstruction.h>
#include <service/event/ctkEvent.h>
#include "ui_SurfaceReconViewWidget.h"
#include <QFuture>
#include <QFutureWatcher>


/**
 * \class SurfaceReconView
 * \brief User interface to provide a reconstructed surface from video images.
 * \ingroup uk_ac_ucl_cmic_igisurfacerecon_internal
*/
class SurfaceReconView : public QmitkBaseView, public Ui::SurfaceReconViewWidget
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  SurfaceReconView();
  virtual ~SurfaceReconView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igisurfacerecon
   */
  static const std::string VIEW_ID;

  /**
   * \brief Returns the view ID.
   */

  virtual std::string GetViewID() const;

protected:

  /**
   *  \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent);

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus();

protected slots:

  /**
   * \brief The main method to perform the surface reconstruction.
   */
  void DoSurfaceReconstruction();

private slots:
  
  /**
   * \brief We can listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

  // we connect the future to this slot
  void OnBackgroundProcessFinished();

private:

  /**
   * \brief Retrieve's the pref values from preference service, and stored in member variables.
   */
  void RetrievePreferenceValues();

  /**
   * \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
   */
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /**
   * \brief Delegate all functionality to this class, so we can unit test it outside of the plugin.
   */
  niftk::SurfaceReconstruction::Pointer      m_SurfaceReconstruction;


  QFuture<mitk::BaseData::Pointer>           m_BackgroundProcess;
  QFutureWatcher<mitk::BaseData::Pointer>    m_BackgroundProcessWatcher;
  std::string                                m_BackgroundOutputNodeName;
  std::string                                m_BackgroundLeftNodeName;
  std::string                                m_BackgroundRightNodeName;
};

#endif // SurfaceReconView_h
