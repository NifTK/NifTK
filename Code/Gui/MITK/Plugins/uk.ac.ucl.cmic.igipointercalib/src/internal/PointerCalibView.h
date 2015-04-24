/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef PointerCalibView_h
#define PointerCalibView_h

#include "ui_PointerCalibView.h"
#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkPointSet.h>
#include <mitkPointSetDataInteractor.h>
#include <mitkUltrasoundPointerBasedCalibration.h>

/**
 * \class PointerCalibView
 * \brief User interface to provide controls for the Ultrasound Pointer Calibration View.
 * \ingroup uk_ac_ucl_cmic_igipointercalib_internal
*/
class PointerCalibView : public QmitkBaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  PointerCalibView();
  virtual ~PointerCalibView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igipointercalib
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

protected:

private slots:

  /**
   * \brief We listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

  void OnSaveToFileButtonPressed();

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
   * \brief UpdateDisplayedPoints will repopulate the displayed list of points.
   */
  void UpdateDisplayedPoints();

  /**
   * \brief If we have > 3 points, we update registration.
   */
  void UpdateRegistration();

  /**
   * \brief Callback for when an Ultrasound image point is added.
   */
  void OnPointAdded();

  /**
   * \brief Callback for when an Ultrasound image point is removed.
   */
  void OnPointRemoved();

  /**
   * \brief Returns the pointer tip in the coordinate frame of the sensor attached to the probe.
   */
  mitk::Point3D GetPointerTipInSensorCoordinates() const;

  /**
   * \brief Checks the size of a and b differs by 1, then finds the missing point id.
   */
  mitk::PointSet::PointIdentifier GetMissingPointId(const mitk::PointSet::Pointer& a,
                                                    const mitk::PointSet::Pointer& b);

  /**
   * \brief All the controls for the main view part.
   */
  Ui::PointerCalibView *m_Controls;

  /**
   * \brief Member variables for keeping state between button clicks.
   */
  mitk::DataStorage::Pointer                       m_DataStorage;
  mitk::UltrasoundPointerBasedCalibration::Pointer m_Calibrator;
  mitk::PointSet::Pointer                          m_ImagePoints;
  mitk::DataNode::Pointer                          m_ImagePointsNode;
  mitk::PointSet::Pointer                          m_SensorPoints;
  mitk::PointSetDataInteractor::Pointer            m_Interactor;
  long                                             m_ImagePointsAddObserverTag;
  long                                             m_ImagePointsRemoveObserverTag;
};

#endif // PointerCalibView_h
