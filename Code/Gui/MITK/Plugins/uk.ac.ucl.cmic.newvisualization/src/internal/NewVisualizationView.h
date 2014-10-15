/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef NewVisualizationView_h
#define NewVisualizationView_h

#include "ui_NewVisualizationViewControls.h"

#include <berryQtViewPart.h>
#include <berryIBerryPreferences.h>

#include <QmitkBaseView.h>
#include <QmitkRenderWindow.h>

#include <mitkDataNode.h>
#include <mitkOCLResourceService.h>

#include <e:\Niftike-r\VL-src\src\examples\Applets\App_VolumeSliced.cpp>

/**
 * \class NewVisualizationView
 * \brief Provides a simple GUI to visualize risk associate for a trajectory
 * \ingroup uk_ac_ucl_cmic_NewVisualization_internal
 */
class NewVisualizationView : public QmitkBaseView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:
  explicit NewVisualizationView();
  virtual ~NewVisualizationView();

  static const std::string VIEW_ID;


protected slots:

protected:

  virtual void CreateQtPartControl(QWidget *parent);

  virtual void SetFocus();

    /// \brief called by QmitkFunctionality when DataManager's selection has changed
  virtual void OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                     const QList<mitk::DataNode::Pointer>& nodes );

  /// \brief Called by framework when a node was modified in the datastorage
  virtual void NodeChanged(const mitk::DataNode* node);

  /// \brief Called by framework when a node was removed from the datastorage
  virtual void NodeRemoved(const mitk::DataNode* node);

  /// \brief Called by framework when a node was added to the datastorage
  virtual void NodeAdded(const mitk::DataNode* node);

private slots: 


private: 
  /// \brief 
  void UpdateDisplay(bool viewEnabled = true);

  /// \brief All the controls for the main view part.
  Ui::NewVisualizationViewControls* m_Controls;

  /// \brief Store a reference to the parent widget of this view.
  QWidget *m_Parent;
  QmitkRenderWindow * m_RenderWindow;

  vl::ref<App_VolumeSliced> m_RenderApplet;
};

#endif // NewVisualizationView_h
