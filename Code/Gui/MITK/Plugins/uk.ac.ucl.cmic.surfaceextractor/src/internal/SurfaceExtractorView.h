/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _SurfaceExtractorView_h
#define _SurfaceExtractorView_h

#include "ui_SurfaceExtractorViewControls.h"

#include <berryQtViewPart.h>
#include <berryIBerryPreferences.h>
#include <QmitkAbstractView.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>

class QWidget;
class SurfaceExtractorViewPrivate;

/**
 * \class SurfaceExtractorView
 * \brief Provides a simple GUI to extract the surface of 3D volumes.
 * \ingroup uk_ac_ucl_cmic_surfaceextractor_internal
 */
class SurfaceExtractorView : public QmitkAbstractView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  explicit SurfaceExtractorView();
  virtual ~SurfaceExtractorView();

  /// \brief Each view for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes);

protected slots:

private slots:

  void on_btnCreate_clicked();
  void on_btnUpdate_clicked();
  void on_btnCancel_clicked();
  void on_cbxGaussianSmooth_toggled(bool checked);

private:

  /// \brief Creation of the connections of widgets to slots.
  void CreateConnections();

  /// \brief Retrieve's the pref values from preference service, and stored in member variables.
  void RetrievePreferenceValues();

  /// \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  void selectReferenceNode(mitk::DataNode::Pointer node);
  void selectSurfaceNode(mitk::DataNode::Pointer node);
  void deselectNode();

  void loadParameters();
  void saveParameters();
  void updateFields();

  /// \brief Enables/Disables controls.
  void EnableControls(bool b);

  /// \brief Returns the derived node that contains the surface.
  mitk::DataStorage::SetOfObjects::ConstPointer findSurfaceNodesOf(mitk::DataNode::Pointer referenceNode);

  /// \brief Returns the source node that this surface node belongs to.
  mitk::DataNode::Pointer findReferenceNodeOf(mitk::DataNode::Pointer surfaceNode);

  void createSurfaceNode();
  void updateSurfaceNode();

  /// \brief All the controls for the main view part.
  Ui::SurfaceExtractorViewControls* m_Controls;

  // Store a reference to the parent widget of this view.
  QWidget *m_Parent;

  QScopedPointer<SurfaceExtractorViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(SurfaceExtractorView);
  Q_DISABLE_COPY(SurfaceExtractorView);
};
#endif // _SurfaceExtractorView_h
