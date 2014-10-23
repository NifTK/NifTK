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
#include <mitkOclResourceService.h>
#include <mitkSurface.h>
#include <mitkImage.h>

#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkDataNodePropertyListener.h>

// VL includes
#include <vlCore/VisualizationLibrary.hpp>
#include <vlQt4/Qt4Widget.hpp>

#include "VLRenderingApplet.h"

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
  /// \brief Slider moved
  void On_SliderMoved(int val);

protected:

  virtual void CreateQtPartControl(QWidget *parent);

  virtual void SetFocus();

  /// \brief Called by framework when a node was removed from the datastorage
  virtual void OnNodeRemoved(mitk::DataNode* node);

  /// \brief Called by framework when a node was added to the datastorage
  virtual void OnNodeAdded(mitk::DataNode* node);
  virtual void OnNodeDeleted(mitk::DataNode* node);

  void OnNamePropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);
  void OnVisibilityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);
  void OnColorPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);
  void OnOpacityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);

private slots: 


private: 
  void InitVLRendering();

  /// \brief 
  //void UpdateDisplay(bool viewEnabled = true);

  /// \brief All the controls for the main view part.
  Ui::NewVisualizationViewControls* m_Controls;

  /// \brief Store a reference to the parent widget of this view.
  QWidget *m_Parent;

  vl::ref<vlQt4::Qt4Widget>  m_VLQtRenderWindow;
  vl::ref<VLRenderingApplet> m_RenderApplet;

  
  mitk::DataNodePropertyListener::Pointer    m_SelectionListener;
  mitk::DataNodePropertyListener::Pointer    m_VisibilityListener;
  mitk::DataNodePropertyListener::Pointer    m_NamePropertyListener;
  mitk::DataNodePropertyListener::Pointer    m_ColorPropertyListener;
  mitk::DataNodePropertyListener::Pointer    m_OpacityPropertyListener;
};

#endif // NewVisualizationView_h
