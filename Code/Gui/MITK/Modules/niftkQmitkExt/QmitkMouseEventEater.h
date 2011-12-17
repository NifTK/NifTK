/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 16:50:16 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7860 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMOUSEEVENTEATER_H_
#define QMITKMOUSEEVENTEATER_H_

#include "niftkQmitkExtExports.h"

#include <QWidget>
#include <QEvent>

/**
 *
 */
class NIFTKQMITKEXT_EXPORT QmitkMouseEventEater : public QObject
{
  Q_OBJECT

public:
  QmitkMouseEventEater(QWidget* parent=NULL) : QObject(parent) { m_IsEating = true; }
  ~QmitkMouseEventEater() {};
  void SetIsEating(bool b) { m_IsEating = b; }
  bool GetIsEating() const { return m_IsEating; }
 protected:
  virtual bool eventFilter(QObject *obj, QEvent *event)
  {
    if (m_IsEating &&
        (   event->type() == QEvent::MouseButtonDblClick
         || event->type() == QEvent::MouseButtonPress
         || event->type() == QEvent::MouseButtonRelease
         || event->type() == QEvent::MouseMove
         || event->type() == QEvent::MouseTrackingChange
        )) {
      return true;
    } else {
      // standard event processing
      return QObject::eventFilter(obj, event);
    }
  }
 private:
  bool m_IsEating;
 }; // end class

#endif
