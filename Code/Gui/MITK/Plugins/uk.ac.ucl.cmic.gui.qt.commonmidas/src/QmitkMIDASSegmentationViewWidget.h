/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_QmitkMIDASSegmentationViewWidget_h
#define mitk_QmitkMIDASSegmentationViewWidget_h

#include "ui_QmitkMIDASSegmentationViewWidget.h"
#include <uk_ac_ucl_cmic_gui_qt_commonmidas_Export.h>
#include <QWidget>
#include <QString>
#include <mitkMIDASEnums.h>
#include <mitkMIDASNodeAddedVisibilitySetter.h>
#include <mitkDataStorageVisibilityTracker.h>

namespace mitk
{
class DataStorage;
class BaseRenderer;
}

class QmitkMIDASSingleViewWidgetListVisibilityManager;
class QmitkMIDASBaseSegmentationFunctionality;
class QmitkRenderWindow;

/**
 * \class QmitkMIDASSegmentationViewWidget
 * \brief Qt Widget to provide a single QmitkMIDASSingleViewWidget, and some associated
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
class CMIC_QT_COMMONMIDAS QmitkMIDASSegmentationViewWidget :
  public QWidget,
  public Ui_QmitkMIDASSegmentationViewWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  QmitkMIDASSegmentationViewWidget(QWidget* parent = 0);
  virtual ~QmitkMIDASSegmentationViewWidget();

  /**
   * \brief Injects the data storage, which is passed onto the contained QmitkMIDASSingleViewWidget.
   * \param storage The data storage for this widget to used, normally taken from the default data storage for the app.
   */
  void SetDataStorage(mitk::DataStorage* storage);

  /**
   * \brief Sets the containing functionality for callback purposes.
   *
   * The reason we do this, is so that we can ask QmitkAbstractView for the mitkIRenderWindowPart
   * rather than have any hard coded reference to any widget such as QmitkMIDASStdMultiWidget.
   *
   * \param functionality In old terminology, the "functionality" that contains this widget,
   * is the child of QmitkAbstractView that contains this widget.
   *
   * \see QmitkMIDASBaseSegmentationFunctionality
   * \see QmitkAbstractView
   */
  void SetContainingFunctionality(QmitkMIDASBaseSegmentationFunctionality* functionality);

  /**
   * \brief Calls setEnabled(enabled) on all contained GUI widgets, except the QmitkMIDASSingleViewWidget.
   * \param enabled if true will enable all widgets, and if false will disable them.
   */
  void SetEnabled(bool enabled);

  /**
   * \brief Connects the widget to the FocusManager.
   */
  void Activated();

  /**
   * \brief Disconnects the widget from the FocusManager.
   */
  void Deactivated();

signals:

  /**
   * \brief At the moment we support single axial, coronal, or sagittal render windows, or combinations of
   * two render windows, in vertical or horizontal mode and ortho view (see MIDASLayout enum for a complete list),
   * and emit this signal when the displayed layout of this window changes.
   */
  void LayoutChanged(MIDASLayout);

protected slots:

  /// \brief Called when any of the layout radio buttons is toggled.
  void OnLayoutRadioButtonToggled(bool checked);

  /// \brief Called when the window layout is selected in the the combo box.
  void OnMultiWindowComboBoxIndexChanged();

  /// \brief Called when the magnification is changed by the spin box.
  void OnMagnificationChanged(double magnification);

  /// \brief Called when the magnification is changed by zooming in a renderer window.
  void OnScaleFactorChanged(QmitkMIDASSingleViewWidget* view, double magnification);

protected:

private:

  /// \brief Method that actually changes the layout to axial, sagittal, coronal etc.
  void ChangeLayout(bool isInitialising = false);

  /// \brief Callback for when the focus changes, where we update the geometry to match the right window.
  void OnFocusChanged();

  /// \brief Works out the MIDASOrientation of the currently focused window.
  MIDASOrientation GetCurrentMainWindowOrientation();

  /// \brief Works out the MIDASLayout of the currently focused window.
  MIDASLayout GetCurrentMainWindowLayout();

  QmitkMIDASBaseSegmentationFunctionality* m_ContainingFunctionality;
  unsigned long m_FocusManagerObserverTag;

  /// \brief Stores the currently selected window layout.
  MIDASLayout m_Layout;

  MIDASLayout m_MainWindowLayout;

  QmitkRenderWindow* m_MainWindowAxial;
  QmitkRenderWindow* m_MainWindowSagittal;
  QmitkRenderWindow* m_MainWindowCoronal;
  QmitkRenderWindow* m_MainWindow3d;
  mitk::BaseRenderer* m_CurrentRenderer;

  mitk::MIDASNodeAddedVisibilitySetter::Pointer m_NodeAddedSetter;
  mitk::DataStorageVisibilityTracker::Pointer m_VisibilityTracker;

  double m_Magnification;
};

#endif
