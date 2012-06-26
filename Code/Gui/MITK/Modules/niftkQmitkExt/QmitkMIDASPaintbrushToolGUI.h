/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMIDASPAINTBRUSHTOOL_H
#define QMITKMIDASMIDASPAINTBRUSHTOOL_H

#include "QmitkToolGUI.h"
#include "mitkMIDASPaintbrushTool.h"
#include <niftkQmitkExtExports.h>

class QSlider;
class QLabel;
class QFrame;

/**
 * \class QmitkMIDASPaintbrushToolGUI
 * \brief GUI component for the mitk::MIDASPaintbrushTool, providing the number of pixels in radius for
 * the cursor.
 *
 * Notice how this class can have a reference to the mitk::Tool it is controlling, and registers with the
 * mitk::Tool in the OnNewToolAssociated method, and de-registers with the mitk::Tool in the destructor.
 *
 * The reverse is not true. Any mitk::Tool must not know that it has a GUI, and hence the reason they
 * are in a different library / Module.
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASPaintbrushToolGUI : public QmitkToolGUI
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkMIDASPaintbrushToolGUI, QmitkToolGUI);
  itkNewMacro(QmitkMIDASPaintbrushToolGUI);

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

  QmitkMIDASPaintbrushToolGUI();
  virtual ~QmitkMIDASPaintbrushToolGUI();

  QSlider* m_Slider;
  QLabel* m_SizeLabel;
  QFrame* m_Frame;

  mitk::MIDASPaintbrushTool::Pointer m_PaintbrushTool;
};

#endif

