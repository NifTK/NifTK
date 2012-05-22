/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-02-16 21:02:48 +0000 (Thu, 16 Feb 2012) $
 Revision          : $Revision: 8525 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMIDASDRAWTOOL_H
#define QMITKMIDASMIDASDRAWTOOL_H

#include "QmitkToolGUI.h"
#include "mitkMIDASDrawTool.h"
#include "niftkQmitkExtExports.h"

class QSlider;
class QLabel;
class QFrame;

/**
 * \class QmitkMIDASDrawToolGUI
 * \brief GUI component for the mitk::MIDASDrawTool
*/
class NIFTKQMITKEXT_EXPORT QmitkMIDASDrawToolGUI : public QmitkToolGUI
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkMIDASDrawToolGUI, QmitkToolGUI);
  itkNewMacro(QmitkMIDASDrawToolGUI);
  void OnCursorSizeChanged(int current);

signals:

public slots:

protected slots:

  void OnNewToolAssociated(mitk::Tool*);
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

