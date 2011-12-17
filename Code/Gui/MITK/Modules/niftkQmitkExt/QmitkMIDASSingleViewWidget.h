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

class QGridLayout;
class QmitkMIDASRenderWindow;

/**
 * \class QmitkMIDASSingleViewWidget
 * \brief A widget to take care of a single QmitkMIDASRenderWindow view,
 * providing a border, background, and methods for remembering the last slice,
 * magnification and position in any given orientation.
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

  QmitkMIDASSingleViewWidget(QWidget *parent, QString windowName);
  ~QmitkMIDASSingleViewWidget();

  /// \brief Sets all the contents margins on the contained layout.
  void SetContentsMargins(int margin);

  /// \brief Sets all the spacing on the contained layout.
  void SetSpacing(int spacing);

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

  /// \brief Disconnects the QmitkMIDASRenderWindow from the mitk::RenderingManager.
  void RemoveFromRenderingManager();

  /// \brief Connects the QmitkMIDASRenderWindow to the mitk::RenderingManager.
  void AddToRenderingManager();

  /// \brief Stores a pointer to the provided geometry, and resets the previous slice number, magnification factor and orientation fields.
  void InitializeGeometry(mitk::Geometry3D::Pointer geometry);

  /// \brief Sets the view orientation to either axial, sagittal or coronal.
  void SetViewOrientation(MIDASViewOrientation orientation);

  /// \brief Get the view orientation.
  MIDASViewOrientation GetViewOrientation() const;

  /// \brief Set the current slice number.
  void SetSliceNumber(int sliceNumber);

  /// \brief Get the current slice number.
  int GetSliceNumber() const;

  /// \brief Set the current magnification factor.
  void SetMagnificationFactor(int magnificationFactor);

  /// \brief Get the current magnification factor.
  int GetMagnificationFactor() const;

  /// \brief Returns the minimum allowed slice number (zero).
  int GetMinSlice() const;

  /// \brief Returns the maximum allowed slice number (number of slices - 1).
  int GetMaxSlice() const;

  /// \brief Returns the minimum allowed magnification (-5)
  int GetMinMagnification() const;

  /// \brief Returns the maximum allowed magnification (20)
  int GetMaxMagnification() const;

private:

  QColor                                         m_BackgroundColor;
  QColor                                         m_SelectedColor;
  QColor                                         m_UnselectedColor;
  QGridLayout                                   *m_Layout;
  QmitkMIDASRenderWindow*                        m_RenderWindow;
  mitk::RenderWindowFrame::Pointer               m_RenderWindowFrame;
  mitk::GradientBackground::Pointer              m_RenderWindowBackground;
  mitk::DataStorage::Pointer                     m_DataStorage;

  int                                            m_SliceNumber;
  int                                            m_MagnificationFactor;
  MIDASViewOrientation                           m_ViewOrientation;
  mitk::Geometry3D*                              m_Geometry;
  std::vector<int>                               m_SliceNumbers;            // length 3
  std::vector<int>                               m_MagnificationFactors;    // length 3
  std::vector<MIDASViewOrientation>              m_Orientations;            // length 3
};

#endif // QMITKMIDASSINGLEVIEWWIDGET_H
