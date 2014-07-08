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

#include <uk_ac_ucl_cmic_sideviewer_Export.h>

#include <berryIPartListener.h>

#include <mitkDataNodeVisibilityTracker.h>
#include <mitkDataNodeStringPropertyFilter.h>
#include <mitkMIDASEnums.h>

#include <QMap>
#include <QWidget>

#include <niftkDnDDisplayEnums.h>

namespace mitk
{
class BaseRenderer;
class DataStorage;
class IRenderWindowPart;
class RenderingManager;
}

class QmitkBaseView;
class QmitkRenderWindow;

class niftkSingleViewerWidget;

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QRadioButton;
class QSpinBox;

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
  public QWidget//,
//  public Ui_QmitkSideViewerWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /// \brief Constructs a QmitkSideViewerWidget object.
  ///
  /// \param view Sets the containing view for callback purposes.
  ///
  ///       The reason we do this, is so that we can ask QmitkAbstractView for the mitkIRenderWindowPart
  ///       rather than have any hard coded reference to any widget such as DnDMultiWindowWidget.
  ///
  QmitkSideViewerWidget(QmitkBaseView* view, QWidget* parent, mitk::RenderingManager* renderingManager);

  /// \brief Destructs the QmitkSideViewerWidget object.
  virtual ~QmitkSideViewerWidget();

  void SetFocused();

  /// \brief Called when the world geometry of main window changes and updates the viewer accordingly.
  void SetGeometry(const itk::EventObject& geometrySendEvent);

  /// \brief Sets the selected render window of the main display.
  /// This view then might need to change its window layout so that it shows the image
  /// of a different orientation.
  /// \param renderWindowPart The render window part (aka. editor or display) that contins the window
  /// \param mainWindow The selected render window of the main display.
  void OnMainWindowChanged(mitk::IRenderWindowPart* renderWindowPart, QmitkRenderWindow* mainWindow);

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
  void OnSelectedPositionChanged(const mitk::Point3D& selectedPosition);

  /// \brief Called when the scale factor is changed by zooming in a renderer window.
  void OnScaleFactorChanged(MIDASOrientation orientation, double scaleFactor);

  /// \brief Called when the window layout of the viewer has changed through interaction.
  void OnWindowLayoutChanged(WindowLayout windowLayout);

private:

  /// \brief Callback for when the focus changes, where we update the geometry to match the right window.
  void OnFocusChanged();

  /// \brief Works out a multi window orientation for the given orientation of the main window.
  WindowLayout GetMultiWindowLayoutForOrientation(MIDASOrientation mainWindowOrientation);

  /// \brief Gets the currently visible editor.
  /// Returns 0 if no editor is opened.
  mitk::IRenderWindowPart* GetSelectedEditor();

  /// \brief Updates the slice and magnification spin boxes according to the selected window.
  void OnViewerWindowChanged();

  /// \brief Method that actually changes the layout to axial, sagittal, coronal etc.
  void OnMainWindowOrientationChanged(MIDASOrientation orientation);

private slots:

  /// \brief Called when one of the main 2D windows has been destroyed.
  virtual void OnAMainWindowDestroyed(QObject* mainWindow);

  /// \brief Fits the displayed region to the size of the viewer window.
  void FitToDisplay();

private:

  void SetupUi(QWidget* parent);

  /// \brief The view that contains this widget.
  QmitkBaseView* m_ContainingView;

  /// \brief The identifier of the focus change listener.
  unsigned long m_FocusManagerObserverTag;

  /// \brief Stores the currently selected window layout.
  WindowLayout m_WindowLayout;

  /// \brief Stores the currently selected window of the visible editor, if any.
  QmitkRenderWindow* m_MainWindow;

  /// \brief The axial window of the selected editor.
  QmitkRenderWindow* m_MainAxialWindow;

  /// \brief The sagittal window of the selected editor.
  QmitkRenderWindow* m_MainSagittalWindow;

  /// \brief The coronal window of the selected editor.
  QmitkRenderWindow* m_MainCoronalWindow;

  /// \brief The slice navigation controller of the currently selected window  of the selected editor.
  mitk::SliceNavigationController* m_MainWindowSnc;

  /// \brief The slice navigation controller of the axial window of the selected editor.
  mitk::SliceNavigationController* m_MainAxialSnc;

  /// \brief The slice navigation controller of the sagittal window of the selected editor.
  mitk::SliceNavigationController* m_MainSagittalSnc;

  /// \brief The slice navigation controller of the coronal window of the selected editor.
  mitk::SliceNavigationController* m_MainCoronalSnc;

  /// \brief Renderer of the currently focused window of the main display.
  mitk::BaseRenderer* m_FocusedRenderer;

  mitk::DataNodeVisibilityTracker::Pointer m_VisibilityTracker;

  /// \brief Filter that tells which nodes should not be handled.
  mitk::DataNodeStringPropertyFilter::Pointer m_MIDASToolNodeNameFilter;

  /// \brief The current magnification in the selected window of the viewer in this widget.
  /// It is used to increase or decrease the magnification value to the closest integers
  /// when using the magnification spin box.
  double m_Magnification;

  /// \brief The orientation of the selected window of the main display.
  MIDASOrientation m_MainWindowOrientation;

  /// \brief Stores the last single window layout of the internal viewer,
  /// one for each orientation of the main window.
  QMap<MIDASOrientation, WindowLayout> m_SingleWindowLayouts;

  /// \brief The world geometry of the selected window of the selected editor.
  /// Any time when the selected main window changes, the world geometry of this viewer
  /// needs to be set to that of the main window.
  const mitk::TimeGeometry* m_TimeGeometry;

  /// \brief Listener to catch events when an editor becomes visible or gets destroyed.
  berry::IPartListener::Pointer m_EditorLifeCycleListener;

  niftkSingleViewerWidget* m_Viewer;
  QWidget* m_ControlsWidget;
  QWidget* m_LayoutWidget;
  QRadioButton* m_CoronalWindowRadioButton;
  QRadioButton* m_SagittalWindowRadioButton;
  QRadioButton* m_AxialWindowRadioButton;
  QRadioButton* m_MultiWindowRadioButton;
  QComboBox* m_MultiWindowComboBox;
  QLabel* m_SliceLabel;
  QSpinBox* m_SliceSpinBox;
  QLabel* m_MagnificationLabel;
  QDoubleSpinBox* m_MagnificationSpinBox;

  mitk::RenderingManager* m_RenderingManager;
};

#endif
