/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 

#ifndef QMITKMIDASSINGLEVIEWWIDGET_H
#define QMITKMIDASSINGLEVIEWWIDGET_H

#include <QWidget>
#include <QColor>
#include "mitkRenderWindowFrame.h"
#include "mitkGradientBackground.h"
#include "mitkDataStorage.h"
#include "mitkTimeSlicedGeometry.h"
#include "mitkSliceNavigationController.h"
#include "mitkRenderingManager.h"

class QGridLayout;
class QmitkMIDASRenderWindow;

/**
 * \class QmitkMIDASSingleViewWidget
 * \brief A widget to take care of a single QmitkMIDASRenderWindow view,
 * providing a border, background, and methods for remembering the last slice,
 * magnification and position in any given orientation.
 *
 * This widget contains its own mitk::RenderingManager and mitk::SliceNavigationControllers.
 * This means it will update and render independently of the rest of the application,
 * and care must be taken to manage this. The reason is that each of these windows could
 * have it's own geometry, and sometimes very different geometry, and when the "Bind Slices"
 * button is clicked, they must all align to a specific (the currently selected Window) geometry.
 * So it became necessary to manage this independent from the rest of the MITK application.
 *
 * <pre>
 * Note: The requirements specification for MIDAS style zoom basically says.
 *
 * magnification   : actual pixels per voxel.
 * on MIDAS widget :
 * 2               : 3
 * 1               : 2
 * 0               : 1 (i.e. no magnification).
 * -1              : 0.5 (i.e. 1 pixel covers 2 voxels).
 * -2              : 0.33 (i.e. 1 pixel covers 3 voxels).
 * etc.
 * </pre>
 *
 * \sa QmitkMIDASRenderWindow
 * \sa QmitkRenderWindow
 */
class QmitkMIDASSingleViewWidget : public QWidget {

  /// \brief Defining Q_OBJECT macro, so we can register signals and slots if needed.
  Q_OBJECT;

public:

  enum MIDASViewOrientation
  {
    MIDAS_VIEW_AXIAL = 0,
    MIDAS_VIEW_SAGITTAL = 1,
    MIDAS_VIEW_CORONAL = 2,
    MIDAS_VIEW_UNKNOWN = 3
  };

  QmitkMIDASSingleViewWidget(QWidget *parent, QString windowName, int minimumMagnification, int maximumMagnification);
  ~QmitkMIDASSingleViewWidget();

  /// \brief Sets all the contents margins on the contained layout.
  void SetContentsMargins(unsigned int margin);

  /// \brief Sets all the spacing on the contained layout.
  void SetSpacing(unsigned int spacing);

  /// \brief Sets whether this widget is considered the selected widget, which means, just setting the border colour.
  void SetSelected(bool selected);

  /// \brief Set the selected color, which defaults to red.
  void SetSelectedColor(QColor color);

  /// \brief Get the selected color, which defaults to red.
  QColor GetSelectedColor() const;

  /// \brief Set the unselected color, which defaults to white.
  void SetUnselectedColor(QColor color);

  /// \brief Get the unselected color, which defaults to white.
  QColor GetUnselectedColor() const;

  /// \brief Set the background color.
  void SetBackgroundColor(QColor color);

  /// \brief Get the background color.
  QColor GetBackgroundColor() const;

  /// \brief Returns the minimum allowed slice number.
  unsigned int GetMinSlice() const;

  /// \brief Returns the maximum allowed slice number.
  unsigned int GetMaxSlice() const;

  /// \brief Returns the minimum allowed time slice number.
  unsigned int GetMinTime() const;

  /// \brief Returns the maximum allowed time slice number.
  unsigned int GetMaxTime() const;

  /// \brief Returns the minimum allowed magnification.
  int GetMinMagnification() const;

  /// \brief Returns the maximum allowed magnification.
  int GetMaxMagnification() const;

  /// \brief Sets the data storage.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief Gets the data storage.
  mitk::DataStorage::Pointer GetDataStorage(mitk::DataStorage* dataStorage);

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsWindow(QmitkMIDASRenderWindow *window) const;

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsVtkRenderWindow(vtkRenderWindow *window) const;

  /// \brief Gets a pointer to the render window.
  QmitkMIDASRenderWindow* GetRenderWindow() const;

  /// \brief Set the current slice number.
  void SetSliceNumber(unsigned int sliceNumber);

  /// \brief Get the current slice number.
  unsigned int GetSliceNumber() const;

  /// \brief Set the current time slice number.
  void SetTime(unsigned int timeSlice);

  /// \brief Get the current time slice number.
  unsigned int GetTime() const;

  /// \brief Set the current magnification factor.
  void SetMagnificationFactor(int magnificationFactor);

  /// \brief Get the current magnification factor.
  int GetMagnificationFactor() const;

  /// \brief Sets the view orientation to either axial, sagittal or coronal.
  void SetViewOrientation(MIDASViewOrientation orientation);

  /// \brief Get the view orientation.
  MIDASViewOrientation GetViewOrientation() const;

  /// \brief Sets the world geometry that we are sampling.
  void SetGeometry(mitk::TimeSlicedGeometry::Pointer geometry);

  /// \brief Gets the world geometry, to pass to other viewers for when slices are bound.
  mitk::TimeSlicedGeometry::Pointer GetGeometry();

  /// \brief Sets the world geometry that we are sampling when we are in bound mode.
  void SetBoundGeometry(mitk::TimeSlicedGeometry::Pointer geometry);

  /// \brief If we tell the widget to be in bound mode, it uses the bound geometries.
  void SetBound(bool isBound);

  /// \brief Returns the bound flag.
  bool GetBound();

  /// \brief As each widget has its own rendering manager, we have to manually ask each widget to re-render.
  void RequestUpdate();

  /// \brief As each widget has its own rendering manager, we have to manually ask each widget to re-render.
  void ForceUpdate();

signals:

  void SliceChanged(QmitkMIDASRenderWindow *window, unsigned int sliceNumber);

protected:

  // overloaded paint handler
  virtual void paintEvent(QPaintEvent* event);

  // Separate method to zoom/magnify the display about the centre of the image. A value > 1 makes the image appear bigger.
  virtual void ZoomDisplayAboutCentre(double scaleFactor);

  // These are both related and use similar calculations, so we calculate them in one method.
  virtual void GetScaleFactors(mitk::Point2D &scaleFactorPixPerVoxel, mitk::Point2D &scaleFactorPixPerMillimetres);

private:

  // Callback for when the slice selector changes slice
  void OnSliceChanged(const itk::EventObject & geometrySliceEvent);

  void ResetRememberedPositions(unsigned int startIndex, unsigned int stopIndex);
  void StorePosition();
  void SetActiveGeometry();
  unsigned int GetBoundUnboundOffset() const;
  unsigned int GetBoundUnboundPreviousArrayOffset() const;

  QmitkMIDASRenderWindow*                        m_RenderWindow;
  mitk::RenderWindowFrame::Pointer               m_RenderWindowFrame;
  mitk::GradientBackground::Pointer              m_RenderWindowBackground;

  mitk::RenderingManager::Pointer                m_RenderingManager;
  mitk::SliceNavigationController::Pointer       m_SliceNavigationController;
  mitk::SliceNavigationController::Pointer       m_TimeNavigationController;

  mitk::DataStorage::Pointer                     m_DataStorage;
  mitk::TimeSlicedGeometry::Pointer              m_UnBoundTimeSlicedGeometry;  // This comes from which ever image is dropped, so not visible outside this class.
  mitk::TimeSlicedGeometry::Pointer              m_BoundTimeSlicedGeometry;    // Passed in, when we do "bind", so shared amongst windows.
  mitk::TimeSlicedGeometry::Pointer              m_ActiveTimeSlicedGeometry;   // The one we use.

  QGridLayout                                   *m_Layout;
  QColor                                         m_BackgroundColor;
  QColor                                         m_SelectedColor;
  QColor                                         m_UnselectedColor;

  int                                            m_MinimumMagnification; // passed in as constructor arguments, so this class unaware of where it came from.
  int                                            m_MaximumMagnification; // passed in as constructor arguments, so this class unaware of where it came from.

  bool                                           m_IsBound;
  bool                                           m_IsSelected;

  std::vector<unsigned int>                      m_CurrentSliceNumbers;          // length 2, one for unbound, then for bound.
  std::vector<unsigned int>                      m_CurrentTimeSliceNumbers;      // length 2, one for unbound, then for bound.
  std::vector<int>                               m_CurrentMagnificationFactors;  // length 2, one for unbound, then for bound.
  std::vector<MIDASViewOrientation>              m_CurrentViewOrientations;      // length 2, one for unbound, then for bound.

  std::vector<unsigned int>                      m_PreviousSliceNumbers;         // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<unsigned int>                      m_PreviousTimeSliceNumbers;     // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<int>                               m_PreviousMagnificationFactors; // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<MIDASViewOrientation>              m_PreviousViewOrientations;     // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.

  // Used for when the slice navigation controller changes slice.
  unsigned long m_SliceSelectorTag;

};

#endif // QMITKMIDASSINGLEVIEWWIDGET_H
