/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkSideViewerWidget_h
#define QmitkSideViewerWidget_h

#include "ui_QmitkSideViewerWidget.h"

#include <uk_ac_ucl_cmic_sideviewer_Export.h>

#include <berryIPartListener.h>

#include <mitkDataNodeAddedVisibilitySetter.h>
#include <mitkDataStorageVisibilityTracker.h>
#include <mitkMIDASDataNodeNameStringFilter.h>
#include <mitkMIDASEnums.h>

#include <QString>
#include <QWidget>

namespace mitk
{
class BaseRenderer;
class DataStorage;
class IRenderWindowPart;
}

class EditorLifeCycleListener;
class QmitkBaseView;
class QmitkRenderWindow;

/**
 * \class QmitkSideViewerWidget
 * \brief Qt Widget to provide a single niftkSingleViewerWidget, and some associated
 * buttons controlling 2/3 view, vertical/horizontal and axial/coronal/sagittal/ortho.
 *
 * The widget will display whatever data nodes are visible in the currently focused
 * render window, not including this widget. This means:
 *
 * <pre>
 * 1. If this widget visible, when new data is added to the data storage, defaults to not-visible in this viewer.
 * 2. When the focus changes, get the current editor axial, sagittal, coronal view:
 *     a. update visibility properties so that whatever is visible in main editor is visible in this widget.
 * </pre>
 */
class CMIC_QT_SIDEVIEWER QmitkSideViewerWidget :
  public QWidget,
  public Ui_QmitkSideViewerWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /**
   * Constructs a QmitkSideViewerWidget object.
   *
   * \param view Sets the containing view for callback purposes.
   *
   *        The reason we do this, is so that we can ask QmitkAbstractView for the mitkIRenderWindowPart
   *        rather than have any hard coded reference to any widget such as DnDMultiWindowWidget.
   */
  QmitkSideViewerWidget(QmitkBaseView* view, QWidget* parent = 0);
  virtual ~QmitkSideViewerWidget();

  /**
   * \brief Injects the data storage, which is passed onto the contained niftkSingleViewerWidget.
   * \param storage The data storage for this widget to used, normally taken from the default data storage for the app.
   */
  void SetDataStorage(mitk::DataStorage* storage);

  /**
   * \brief Calls setEnabled(enabled) on all contained GUI widgets, except the niftkSingleViewerWidget.
   * \param enabled if true will enable all widgets, and if false will disable them.
   */
  void SetEnabled(bool enabled);

  /// \brief Sets the selected render window of the main display.
  /// This view then might need to change its window layout so that it shows the image
  /// of a different orientation.
  /// \param mainWindow The selected render window of the main display.
  void SetMainWindow(QmitkRenderWindow* mainWindow);

signals:

  /**
   * \brief At the moment we support single axial, coronal, or sagittal render windows, or combinations of
   * two render windows, in vertical or horizontal mode and ortho view (see MIDASLayout enum for a complete list),
   * and emit this signal when the displayed layout of this window changes.
   */
  void LayoutChanged(WindowLayout);

protected slots:

  /// \brief Called when the axial window layout radio button is toggled.
  void OnAxialWindowRadioButtonToggled(bool checked);

  /// \brief Called when the sagittal window layout radio button is toggled.
  void OnSagittalWindowRadioButtonToggled(bool checked);

  /// \brief Called when the coronal window layout radio button is toggled.
  void OnCoronalWindowRadioButtonToggled(bool checked);

  /// \brief Called when the multi window layout radio button is toggled.
  void OnMultiWindowRadioButtonToggled(bool checked);

  /// \brief Called when the window layout is selected in the the combo box.
  void OnMultiWindowComboBoxIndexChanged();

  /// \brief Called when the slice is changed by the spin box.
  void OnSliceSpinBoxValueChanged(int slice);

  /// \brief Called when the magnification is changed by the spin box.
  void OnMagnificationSpinBoxValueChanged(double magnification);

  /// \brief Called when the scale factor is changed by zooming in a renderer window.
  void OnSelectedPositionChanged(niftkSingleViewerWidget* viewer, const mitk::Point3D& selectedPosition);

  /// \brief Called when the scale factor is changed by zooming in a renderer window.
  void OnScaleFactorChanged(niftkSingleViewerWidget* viewer, MIDASOrientation orientation, double scaleFactor);

  /// \brief Called when the window layout of the viewer has changed through interaction.
  void OnWindowLayoutChanged(niftkSingleViewerWidget* viewer, WindowLayout windowLayout);

protected:

private:

  /// \brief Method that actually changes the layout to axial, sagittal, coronal etc.
  void ChangeLayout();

  /// \brief Callback for when the focus changes, where we update the geometry to match the right window.
  void OnFocusChanged();

  mitk::IRenderWindowPart* GetSelectedEditor();

  /// \brief Gets the main window that is the last active render window of the main display.
  QmitkRenderWindow* GetMainWindow(const QString& id = QString());

  /// \brief Works out the orientation of the currently focused window.
  MIDASOrientation GetMainWindowOrientation();

  /// \brief Works out the orientation of a renderer.
  MIDASOrientation GetWindowOrientation(mitk::BaseRenderer* renderer);

private slots:

  /// \brief Called when one of the main 2D windows has been destroyed.
  virtual void OnAMainWindowDestroyed(QObject* mainWindow);

private:

  QmitkBaseView* m_ContainingView;
  unsigned long m_FocusManagerObserverTag;

  /// \brief Stores the currently selected window layout.
  WindowLayout m_WindowLayout;

  QmitkRenderWindow* m_MainWindow;

  QmitkRenderWindow* m_MainAxialWindow;
  QmitkRenderWindow* m_MainSagittalWindow;
  QmitkRenderWindow* m_MainCoronalWindow;

  mitk::SliceNavigationController* m_MainAxialSnc;
  mitk::SliceNavigationController* m_MainSagittalSnc;
  mitk::SliceNavigationController* m_MainCoronalSnc;

  /// \brief Renderer of the currently focused window of the main display.
  mitk::BaseRenderer* m_FocusedRenderer;

  mitk::DataNodeAddedVisibilitySetter::Pointer m_NodeAddedSetter;
  mitk::DataStorageVisibilityTracker::Pointer m_VisibilityTracker;

  double m_Magnification;

  /// \brief The orientation of the currently focused window of the main display.
  MIDASOrientation m_MainWindowOrientation;

  /// \brief Stores the last single window layout of the internal viewer,
  /// one for each orientation of the main window.
  QMap<MIDASOrientation, WindowLayout> m_SingleWindowLayouts;

  mitk::MIDASDataNodeNameStringFilter::Pointer m_MIDASToolNodeNameFilter;

  mitk::TimeGeometry* m_Geometry;

  berry::IPartListener::Pointer m_EditorLifeCycleListener;
};

#endif
