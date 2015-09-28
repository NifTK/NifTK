/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef VLRendererView_h
#define VLRendererView_h

#include "ui_VLRendererViewControls.h"

#include <berryQtViewPart.h>
#include <berryIBerryPreferences.h>
#include <QmitkBaseView.h>
#include <QmitkRenderWindow.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <mitkImage.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>


// VL includes
#include <vlCore/VisualizationLibrary.hpp>      // FIXME: should go away!

#include <Rendering/VLQt4Widget.h>


/**
 * \class VLRendererView
 * \brief Provides a simple GUI to visualize stuff
 * \ingroup uk_ac_ucl_cmic_VLRenderer_internal
 */
class VLRendererView : public QmitkBaseView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:
  explicit VLRendererView();
  virtual ~VLRendererView();

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

  virtual void Visible();

private slots: 
  void OnBackgroundNodeSelected(const mitk::DataNode* node);
  void OnCameraNodeSelected(const mitk::DataNode* node);
  void OnCameraNodeEnabled(bool enabled);


private: 
  void InitVLRendering();

  /// \brief 
  void ReinitDisplay(bool viewEnabled = true);

  /// \brief All the controls for the main view part.
  Ui::VLRendererViewControls* m_Controls;

  /// \brief Store a reference to the parent widget of this view.
  QWidget *m_Parent;


  // VL rendering specific members
  // FIXME: should just be a pointer, so we can drop the header dependency on vl.
  vl::ref<VLQt4Widget>       m_VLQtRenderWindow;

  // Listeners
  mitk::DataNodePropertyListener::Pointer    m_SelectionListener;
  mitk::DataNodePropertyListener::Pointer    m_NamePropertyListener;

};

#endif // VLRendererView_h
