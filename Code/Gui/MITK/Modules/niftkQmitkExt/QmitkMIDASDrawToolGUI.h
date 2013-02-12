/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKMIDASMIDASDRAWTOOL_H
#define QMITKMIDASMIDASDRAWTOOL_H

#include "QmitkToolGUI.h"
#include "mitkMIDASDrawTool.h"
#include <niftkQmitkExtExports.h>

class QSlider;
class QLabel;
class QFrame;

/**
 * \class QmitkMIDASDrawToolGUI
 * \brief GUI component for the mitk::MIDASDrawTool, providing a single slider to control the radius in
 * millimetres of the "erase" function.
 *
 * Notice how this class can have a reference to the mitk::Tool it is controlling, and registers with the
 * mitk::Tool in the OnNewToolAssociated method, and de-registers with the mitk::Tool in the destructor.
 *
 * The reverse is not true. Any mitk::Tool must not know that it has a GUI, and hence the reason they
 * are in a different library / Module.
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASDrawToolGUI : public QmitkToolGUI
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkMIDASDrawToolGUI, QmitkToolGUI);
  itkNewMacro(QmitkMIDASDrawToolGUI);

  /// \brief Method to set or initialise the size of the cursor (radius of influence).
  void OnCursorSizeChanged(int current);

signals:

public slots:

protected slots:

  /// \brief Qt slot called when the tool is activated.
  void OnNewToolAssociated(mitk::Tool*);

  /// \brief Qt slot called when the user moves the slider.
  void OnSliderValueChanged(int value);

protected:

  QmitkMIDASDrawToolGUI();
  virtual ~QmitkMIDASDrawToolGUI();

  QSlider* m_Slider;
  QLabel* m_SizeLabel;
  QFrame* m_Frame;

  mitk::MIDASDrawTool::Pointer m_DrawTool;
};

#endif

