/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef TagTrackerView_h
#define TagTrackerView_h

#include "QmitkBaseView.h"
#include <service/event/ctkEvent.h>
#include <mitkDataNode.h>
#include "ui_TagTrackerViewControls.h"
#include <cv.h>

/**
 * \class TagTrackerView
 * \brief User interface to provide a small plugin to track Augmented Reality tags.
 * \ingroup uk_ac_ucl_cmic_igitagtracker_internal
*/
class TagTrackerView : public QmitkBaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  TagTrackerView();
  virtual ~TagTrackerView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igitagtracker
   */
  static const std::string VIEW_ID;

  /**
   * \brief This plugin creates its own data node to store a point set, this static variable stores the name.
   */
  static const std::string NODE_ID;

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
   * \brief We can listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

  /**
   * \brief Or we can listed to a manual update button.
   */
  void OnManualUpdate();

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
   * \brief Loads a matrix.
   */
  void LoadMatrix(const QString& fileName, CvMat *matrixToWriteTo);

  /**
   * \brief Main method to update tag positions.
   */
  void UpdateTags();

  /** The Widgets. */
  Ui::TagTrackerViewControls *m_Controls;

  /** The Member Variables. */
  mitk::DataNode::Pointer m_LeftNode;
  mitk::DataNode::Pointer m_RightNode;
  QString m_LeftIntrinsicFileName;
  QString m_RightIntrinsicFileName;
  QString m_RightToLeftRotationFileName;
  QString m_RightToLeftTranslationFileName;
  CvMat *m_LeftIntrinsicMatrix;
  CvMat *m_RightIntrinsicMatrix;
  CvMat *m_RightToLeftRotationVector;
  CvMat *m_RightToLeftTranslationVector;
  bool m_ListenToEventBusPulse;
  float m_MinSize;
  float m_MaxSize;
};

#endif // TagTrackerView_h
