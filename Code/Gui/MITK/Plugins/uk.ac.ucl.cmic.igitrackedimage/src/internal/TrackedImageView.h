/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef TrackedImageView_h
#define TrackedImageView_h

#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_TrackedImageView.h"
#include <vtkSmartPointer.h>
#include <mitkRenderingManager.h>
#include <mitkVector.h>

class vtkMatrix4x4;

/**
 * \class TrackedImageView
 * \brief User interface to coordinate an image and tracker transform to
 * enable the display of an image in a rendered scene, as might be useful in
 * a tracked ultrasound probe for example.
 * \ingroup uk_ac_ucl_cmic_igitrackedimage_internal
*/
class TrackedImageView : public QmitkBaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  TrackedImageView();
  virtual ~TrackedImageView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igitrackedimage
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

  /**
   * \brief When the image selection is changed, we re-init the views for the tracked image.
   */
  void OnSelectionChanged(const mitk::DataNode*);

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
  Ui::TrackedImageView *m_Controls;

  /**
   * \brief Member variables for keeping state between button clicks.
   */
  vtkSmartPointer<vtkMatrix4x4>   m_ImageToProbeTransform;
  std::string                     m_ImageToProbeFileName;
  mitk::DataNode::Pointer         m_PlaneNode;
  mitk::RenderingManager::Pointer m_RenderingManager;
  mitk::Point2D                   m_ImageScaling;
};

#endif // TrackedImageView_h
