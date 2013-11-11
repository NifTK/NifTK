/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerControls_p_h
#define niftkMultiViewerControls_p_h

#include <QWidget>

#include <niftkDnDDisplayExports.h>

#include "niftkDnDDisplayEnums.h"
#include <niftkSingleViewerControls.h>

#include "ui_niftkMultiViewerControls.h"

//namespace Ui
//{
//class niftkMultiViewerControls;
//}

/**
 * \class niftkMultiViewerControls
 * \brief Control panel for the DnD display.
 */
class NIFTKDNDDISPLAY_EXPORT niftkMultiViewerControls : public niftkSingleViewerControls, private Ui_niftkMultiViewerControls
{
  Q_OBJECT
  
public:

  /// \brief Constructs the niftkMultiViewerControls object.
  explicit niftkMultiViewerControls(QWidget *parent = 0);

  /// \brief Destructs the niftkMultiViewerControls object.
  virtual ~niftkMultiViewerControls();
  
  /// \brief Tells if the multi view controls are visible.
  bool AreViewNumberControlsVisible() const;

  /// \brief Shows or hides the multi view controls.
  void SetViewNumberControlsVisible(bool visible);

  /// \brief Tells if the drop type controls are visible.
  bool AreDropTypeControlsVisible() const;

  /// \brief Shows or hides the drop type controls.
  void SetDropTypeControlsVisible(bool visible);

  /// \brief Tells if the single view controls are enabled.
  bool AreSingleViewControlsEnabled() const;

  /// \brief Enables or disables the single view controls.
  void SetSingleViewControlsEnabled(bool enabled);

  /// \brief Tells if the multi view controls are enabled.
  bool AreMultiViewControlsEnabled() const;

  /// \brief Enables or disables the multi view controls.
  void SetMultiViewControlsEnabled(bool enabled);

  /// \brief Gets the number of rows of the views.
  int GetViewRows() const;

  /// \brief Gets the number of rows of the views.
  int GetViewColumns() const;

  /// \brief Sets the number of the rows and columns of views to the given numbers.
  void SetViewNumber(int rows, int columns);

  /// \brief Gets the maximal number of rows of the views.
  int GetMaxViewRows() const;

  /// \brief Gets the maximal number of columns of the views.
  int GetMaxViewColumns() const;

  /// \brief Sets the maximal number of the rows and columns of views to the given numbers.
  void SetMaxViewNumber(int rows, int columns);

  /// \brief Returns true if the selected position of the views is bound, otherwise false.
  bool AreViewPositionsBound() const;

  /// \brief Sets the bind view positions check box to the given value.
  void SetViewPositionsBound(bool bound);

  /// \brief Returns true if the  cursor of the views is bound, otherwise false.
  bool AreViewCursorsBound() const;

  /// \brief Sets the bind view cursors check box to the given value.
  void SetViewCursorsBound(bool bound);

  /// \brief Returns true if the magnification of the views are bound, otherwise false.
  bool AreViewMagnificationsBound() const;

  /// \brief Sets the bind view magnifications check box to the given value.
  void SetViewMagnificationsBound(bool bound);

  /// \brief Returns true if the  layout of the views is bound, otherwise false.
  bool AreViewLayoutsBound() const;

  /// \brief Sets the bind view layouts check box to the given value.
  void SetViewLayoutsBound(bool bound);

  /// \brief Returns true if the  geometry of the views is bound, otherwise false.
  bool AreViewGeometriesBound() const;

  /// \brief Sets the bind view geometries check box to the given value.
  void SetViewGeometriesBound(bool bound);

  /// \brief Gets the selected drop type.
  DnDDisplayDropType GetDropType() const;

  /// \brief Sets the drop type controls to the given drop type.
  void SetDropType(DnDDisplayDropType dropType);

signals:

  /// \brief Emitted when the selected number of views has been changed.
  void ViewNumberChanged(int rows, int columns);

  /// \brief Emitted when the view position binding option has been changed.
  void ViewPositionBindingChanged(bool bound);

  /// \brief Emitted when the view cursor binding option has been changed.
  void ViewCursorBindingChanged(bool bound);

  /// \brief Emitted when the view magnification binding option has been changed.
  void ViewMagnificationBindingChanged(bool bound);

  /// \brief Emitted when the view layout binding option has been changed.
  void ViewLayoutBindingChanged(bool bound);

  /// \brief Emitted when the view geometry binding option has been changed.
  void ViewGeometryBindingChanged(bool bound);

  /// \brief Emitted when the selected drop type has been changed.
  void DropTypeChanged(DnDDisplayDropType dropType);

  /// \brief Emitted when the drop accumulate option has been changed.
  void DropAccumulateChanged(bool accumulate);

private slots:

  void On1x1ViewsButtonClicked();
  void On1x2ViewsButtonClicked();
  void On1x3ViewsButtonClicked();
  void On2x1ViewsButtonClicked();
  void On2x2ViewsButtonClicked();
  void On2x3ViewsButtonClicked();
  void OnViewRowsSpinBoxValueChanged(int rows);
  void OnViewColumnsSpinBoxValueChanged(int columns);

  void OnViewPositionBindingChanged(bool bound);
  void OnViewCursorBindingChanged(bool bound);

  void OnDropSingleRadioButtonToggled(bool bound);
  void OnDropMultipleRadioButtonToggled(bool bound);
  void OnDropThumbnailRadioButtonToggled(bool bound);

private:

//  Ui_niftkMultiViewerControls* ui;

  bool m_ShowViewNumberControls;
  bool m_ShowDropTypeControls;
};

#endif
