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
  
  /// \brief Tells if the viewer number controls are visible.
  bool AreViewerNumberControlsVisible() const;

  /// \brief Shows or hides the viewer number controls.
  void SetViewerNumberControlsVisible(bool visible);

  /// \brief Tells if the drop type controls are visible.
  bool AreDropTypeControlsVisible() const;

  /// \brief Shows or hides the drop type controls.
  void SetDropTypeControlsVisible(bool visible);

  /// \brief Tells if the single viewer controls are enabled.
  bool AreSingleViewerControlsEnabled() const;

  /// \brief Enables or disables the single viewer controls.
  void SetSingleViewerControlsEnabled(bool enabled);

  /// \brief Tells if the multi viewer controls are enabled.
  bool AreMultiViewerControlsEnabled() const;

  /// \brief Enables or disables the multi viewer controls.
  void SetMultiViewerControlsEnabled(bool enabled);

  /// \brief Gets the number of rows of the viewers.
  int GetViewerRows() const;

  /// \brief Gets the number of rows of the viewers.
  int GetViewerColumns() const;

  /// \brief Sets the number of the rows and columns of viewers to the given numbers.
  void SetViewerNumber(int rows, int columns);

  /// \brief Gets the maximal number of rows of the viewers.
  int GetMaxViewerRows() const;

  /// \brief Gets the maximal number of columns of the viewers.
  int GetMaxViewerColumns() const;

  /// \brief Sets the maximal number of the rows and columns of viewers to the given numbers.
  void SetMaxViewerNumber(int rows, int columns);

  /// \brief Returns true if the selected position of the viewers is bound, otherwise false.
  bool AreViewerPositionsBound() const;

  /// \brief Sets the bind viewer positions check box to the given value.
  void SetViewerPositionsBound(bool bound);

  /// \brief Returns true if the  cursor of the viewers is bound, otherwise false.
  bool AreViewerCursorsBound() const;

  /// \brief Sets the bind viewer cursors check box to the given value.
  void SetViewerCursorsBound(bool bound);

  /// \brief Returns true if the magnification of the viewers are bound, otherwise false.
  bool AreViewerMagnificationsBound() const;

  /// \brief Sets the bind viewer magnifications check box to the given value.
  void SetViewerMagnificationsBound(bool bound);

  /// \brief Returns true if the window layout of the viewers is bound, otherwise false.
  bool AreViewerWindowLayoutsBound() const;

  /// \brief Sets the bind viewer window layouts check box to the given value.
  void SetViewerWindowLayoutsBound(bool bound);

  /// \brief Returns true if the  geometry of the viewers is bound, otherwise false.
  bool AreViewerGeometriesBound() const;

  /// \brief Sets the bind viewer geometries check box to the given value.
  void SetViewerGeometriesBound(bool bound);

  /// \brief Gets the selected drop type.
  DnDDisplayDropType GetDropType() const;

  /// \brief Sets the drop type controls to the given drop type.
  void SetDropType(DnDDisplayDropType dropType);

signals:

  /// \brief Emitted when the selected number of viewers has been changed.
  void ViewerNumberChanged(int rows, int columns);

  /// \brief Emitted when the viewer position binding option has been changed.
  void ViewerPositionBindingChanged(bool bound);

  /// \brief Emitted when the viewer cursor binding option has been changed.
  void ViewerCursorBindingChanged(bool bound);

  /// \brief Emitted when the viewer magnification binding option has been changed.
  void ViewerMagnificationBindingChanged(bool bound);

  /// \brief Emitted when the viewer window layout binding option has been changed.
  void ViewerWindowLayoutBindingChanged(bool bound);

  /// \brief Emitted when the viewer geometry binding option has been changed.
  void ViewerGeometryBindingChanged(bool bound);

  /// \brief Emitted when the selected drop type has been changed.
  void DropTypeChanged(DnDDisplayDropType dropType);

  /// \brief Emitted when the drop accumulate option has been changed.
  void DropAccumulateChanged(bool accumulate);

private slots:

  void On1x1ViewerButtonClicked();
  void On1x2ViewersButtonClicked();
  void On1x3ViewersButtonClicked();
  void On2x1ViewersButtonClicked();
  void On2x2ViewersButtonClicked();
  void On2x3ViewersButtonClicked();
  void OnViewerRowsSpinBoxValueChanged(int rows);
  void OnViewerColumnsSpinBoxValueChanged(int columns);

  void OnViewerPositionBindingChanged(bool bound);
  void OnViewerCursorBindingChanged(bool bound);

  void OnDropSingleRadioButtonToggled(bool bound);
  void OnDropMultipleRadioButtonToggled(bool bound);
  void OnDropThumbnailRadioButtonToggled(bool bound);

private:

  bool m_ShowViewerNumberControls;
  bool m_ShowDropTypeControls;
};

#endif
