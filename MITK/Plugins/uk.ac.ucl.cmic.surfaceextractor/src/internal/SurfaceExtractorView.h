/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceExtractorView_h
#define SurfaceExtractorView_h

#include "ui_SurfaceExtractorViewControls.h"

#include <berryQtViewPart.h>
#include <berryIBerryPreferences.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkImageToSurfaceFilter.h>

#include <niftkBaseView.h>

class QWidget;
class QEvent;
class SurfaceExtractorViewPrivate;

/**
 * \class SurfaceExtractorView
 * \brief Provides a simple GUI to extract the surface of 3D volumes.
 * \ingroup uk_ac_ucl_cmic_surfaceextractor_internal
 */
class SurfaceExtractorView : public niftk::BaseView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /**
   * \brief Each View for a plugin has its own globally unique ID, this one is
   * "uk.ac.ucl.cmic.surfaceextractor" and the .cxx file and plugin.xml should match.
   */
  static const QString VIEW_ID;

  explicit SurfaceExtractorView();
  virtual ~SurfaceExtractorView();

protected:

  /**
   * \see mitk::ILifecycleAwarePart::PartActivated
   */
  virtual void Activated() override;

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent) override;

  virtual bool eventFilter(QObject *obj, QEvent *event) override;

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus() override;

  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes) override;

protected slots:

private slots:

  void OnApplyClicked();
  void OnAdvancedFeaturesToggled(int state);
  void OnExtractionMethodChanged(int which);

private:

  /// \brief Retrieve's the pref values from preference service, and stored in member variables.
  void RetrievePreferenceValues();

  /// \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*) override;

  void SelectReferenceNode(mitk::DataNode::Pointer node);
  void SelectSurfaceNode(mitk::DataNode::Pointer node);
  void DeselectNode();

  void LoadParameters();
  void SaveParameters();
  void UpdateFields();

  /// \brief Enables/Disables controls.
  void EnableControls(bool b);

  /// \brief Returns the derived node that contains the surface.
  mitk::DataStorage::SetOfObjects::ConstPointer findSurfaceNodesOf(mitk::DataNode::Pointer referenceNode);

  /// \brief Returns the source node that this surface node belongs to.
  mitk::DataNode::Pointer findReferenceNodeOf(mitk::DataNode::Pointer surfaceNode);

  void CreateSurfaceNode();
  void UpdateSurfaceNode();

private:
  /// \brief All the controls for the main view part.
  Ui::SurfaceExtractorViewControls* m_Controls;

  // Store a reference to the parent widget of this view.
  QWidget *m_Parent;

  QScopedPointer<SurfaceExtractorViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(SurfaceExtractorView);
  Q_DISABLE_COPY(SurfaceExtractorView);
};
#endif // _SurfaceExtractorView_h
