/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKMOUSEEVENTEATER_H_
#define QMITKMOUSEEVENTEATER_H_

#include "niftkCoreGuiExports.h"
#include <QWidget>
#include <QEvent>

/**
 * \class QmitkMouseEventEater
 * \brief Qt event filter to eat mouse events
 * \ingroup uk.ac.ucl.cmic.thumbnail
 */
class NIFTKCOREGUI_EXPORT QmitkMouseEventEater : public QObject
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
