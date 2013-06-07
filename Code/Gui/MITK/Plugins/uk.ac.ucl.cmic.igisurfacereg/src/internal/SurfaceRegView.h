/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef SurfaceRegView_h
#define SurfaceRegView_h

#include <QmitkBaseView.h>
#include "ui_SurfaceRegView.h"
#include <vtkSmartPointer.h>
#include <mitkSurfaceBasedRegistration.h>

class vtkMatrix4x4;

/**
 * \class SurfaceRegView
 * \brief User interface to provide controls for surface based registration.
 *
 * This class manages user interaction, but delegates the algorithm to
 * mitk::SurfaceBasedRegistration.
 *
 * \ingroup uk_ac_ucl_cmic_igisurfacereg_internal
*/
class SurfaceRegView : public QmitkBaseView
{  
  /**
   * this is needed for all Qt objects that should have a Qt meta-object
   * (everything that derives from QObject and wants to have signal/slots)
   */
  Q_OBJECT

public:

  SurfaceRegView();
  virtual ~SurfaceRegView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igisurfacereg
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

  void OnCalculateButtonPressed();
  void OnComposeWithDataButtonPressed();
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
   * \brief All the controls for the main view part.
   */
  Ui::SurfaceRegView *m_Controls;
  vtkSmartPointer<vtkMatrix4x4> m_Matrix;

  int m_MaxIterations;
  int m_MaxPoints;
  mitk::SurfaceBasedRegistration::Method m_Method;

};

#endif // SurfaceRegView_h
