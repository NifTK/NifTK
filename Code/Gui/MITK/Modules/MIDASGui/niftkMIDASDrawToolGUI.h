/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMIDASDrawToolGUI_h
#define niftkMIDASDrawToolGUI_h

#include <niftkMIDASGuiExports.h>
#include <niftkMIDASDrawTool.h>
#include <QmitkToolGUI.h>

class QFrame;

class ctkSliderWidget;

/**
 * \class niftkMIDASDrawToolGUI
 * \brief GUI component for the niftk::MIDASDrawTool, providing a single slider to control the radius in
 * millimetres of the "erase" function.
 *
 * Notice how this class can have a reference to the mitk::Tool it is controlling, and registers with the
 * mitk::Tool in the OnNewToolAssociated method, and de-registers with the mitk::Tool in the destructor.
 *
 * The reverse is not true. Any mitk::Tool must not know that it has a GUI, and hence the reason they
 * are in a different library / Module.
 */
class NIFTKMIDASGUI_EXPORT niftkMIDASDrawToolGUI : public QmitkToolGUI
{
  Q_OBJECT

public:

  mitkClassMacro(niftkMIDASDrawToolGUI, QmitkToolGUI);
  itkNewMacro(niftkMIDASDrawToolGUI);

  /// \brief Method to set or initialise the size of the cursor (radius of influence).
  void OnCursorSizeChanged(double cursorSize);

signals:

public slots:

protected slots:

  /// \brief Qt slot called when the tool is activated.
  void OnNewToolAssociated(mitk::Tool*);

  /// \brief Qt slot called when the user moves the slider.
  void OnSliderValueChanged(double value);

protected:

  niftkMIDASDrawToolGUI();
  virtual ~niftkMIDASDrawToolGUI();

  ctkSliderWidget* m_Slider;
  QFrame* m_Frame;

  niftk::MIDASDrawTool::Pointer m_DrawTool;
};

#endif

