/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPaintbrushToolGUI_h
#define niftkPaintbrushToolGUI_h

#include <QmitkToolGUI.h>
#include <niftkPaintbrushTool.h>

class QFrame;
class QLabel;
class QTimer;

class ctkSliderWidget;

namespace niftk
{

/// \class niftkPaintbrushToolGUI
/// \brief GUI component for the PaintbrushTool, providing the number of pixels in radius for
/// the eraser.
///
/// Notice how this class can have a reference to the mitk::Tool it is controlling, and registers with the
/// mitk::Tool in the OnNewToolAssociated method, and de-registers with the mitk::Tool in the destructor.
///
/// The reverse is not true. Any mitk::Tool must not know that it has a GUI, and hence the reason they
/// are in a different library / Module.
class PaintbrushToolGUI : public QmitkToolGUI
{
  Q_OBJECT

public:

  mitkClassMacro(PaintbrushToolGUI, QmitkToolGUI)
  itkNewMacro(PaintbrushToolGUI)

  /// \brief Method to set or initialise the size of the eraser (radius of influence).
  void OnEraserSizeChangedInTool(double eraserSize);

signals:

public slots:

protected slots:

  /// \brief Qt slot called when the tool is activated.
  void OnNewToolAssociated(mitk::Tool*);

  /// \brief Qt slot called when the user moves the slider.
  void OnEraserSizeChangedInGui(double value);

  /// \brief Qt slot called after the user stopped dragging the slider.
  void OnSettingEraserSizeFinished();

protected:

  PaintbrushToolGUI();
  virtual ~PaintbrushToolGUI();

  ctkSliderWidget* m_Slider;
  QLabel* m_SizeLabel;
  QFrame* m_Frame;

  PaintbrushTool::Pointer m_PaintbrushTool;

private:

  QTimer* m_ShowEraserTimer;

};

}

#endif

