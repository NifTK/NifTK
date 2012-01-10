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
#include "mitkGeometry3D.h"
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

  /// \brief Disconnects the QmitkMIDASRenderWindow from the mitk::RenderingManager.
  void RemoveFromRenderingManager();

  /// \brief Connects the QmitkMIDASRenderWindow to the mitk::RenderingManager.
  void AddToRenderingManager();

  /// \brief Sets the view orientation to either axial, sagittal or coronal.
  void SetViewOrientation(MIDASViewOrientation orientation);

  /// \brief Get the view orientation.
  MIDASViewOrientation GetViewOrientation() const;

  /// \brief Stores a pointer to the provided geometry, and resets the previous slice number, magnification factor and orientation fields.
  void InitializeGeometry(mitk::Geometry3D::Pointer geometry);

private:

  QmitkMIDASRenderWindow*                        m_RenderWindow;
  mitk::RenderWindowFrame::Pointer               m_RenderWindowFrame;
  mitk::GradientBackground::Pointer              m_RenderWindowBackground;

  mitk::RenderingManager::Pointer                m_RenderingManager;
  mitk::SliceNavigationController::Pointer       m_SliceNavigationController;

  mitk::DataStorage::Pointer                     m_DataStorage;
  mitk::Geometry3D*                              m_Geometry;

  QGridLayout                                   *m_Layout;
  QColor                                         m_BackgroundColor;
  QColor                                         m_SelectedColor;
  QColor                                         m_UnselectedColor;

  int                                            m_MinimumMagnification; // passed in as constructor arguments, so this class unaware.
  int                                            m_MaximumMagnification; // passed in as constructor arguments, so this class unaware.

  unsigned int                                   m_SliceNumber;
  unsigned int                                   m_TimeSliceNumber;
  int                                            m_MagnificationFactor;
  MIDASViewOrientation                           m_ViewOrientation;

  std::vector<unsigned int>                      m_SliceNumbers;            // length 3, one each for axial, sagittal, coronal.
  std::vector<unsigned int>                      m_TimeSliceNumbers;        // length 3, one each for axial, sagittal, coronal.
  std::vector<int>                               m_MagnificationFactors;    // length 3, one each for axial, sagittal, coronal.
  std::vector<MIDASViewOrientation>              m_ViewOrientations;        // length 3, one each for axial, sagittal, coronal.
};

#endif // QMITKMIDASSINGLEVIEWWIDGET_H
