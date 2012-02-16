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
#include "niftkQmitkExtExports.h"

class QSlider;
class QLabel;
class QFrame;

/**
 * \class QmitkMIDASPaintbrushToolGUI
 * \brief GUI component for the mitkMIDASPaintbrushTool
*/
class NIFTKQMITKEXT_EXPORT QmitkMIDASPaintbrushToolGUI : public QmitkToolGUI
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkMIDASPaintbrushToolGUI, QmitkToolGUI);
  itkNewMacro(QmitkMIDASPaintbrushToolGUI);
  void OnCursorSizeChanged(int current);

signals:

public slots:

protected slots:

  void OnNewToolAssociated(mitk::Tool*);
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

