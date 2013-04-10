/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKPAINTEVENTEATER_H_
#define QMITKPAINTEVENTEATER_H_

#include "niftkCoreGuiExports.h"
#include <QWidget>
#include <QEvent>

/**
 * \class QmitkPaintEventEater
 * \brief Qt Event Filter to eat paint events.
 */
class NIFTKCOREGUI_EXPORT QmitkPaintEventEater : public QObject
{
  Q_OBJECT

public:
  QmitkPaintEventEater(QWidget* parent=NULL) : QObject(parent) { m_IsEating = true; }
  ~QmitkPaintEventEater() {};
  void SetIsEating(bool b) { m_IsEating = b; }
  bool GetIsEating() const { return m_IsEating; }
 protected:
  virtual bool eventFilter(QObject *obj, QEvent *event)
  {
    if (m_IsEating && event->type() == QEvent::Paint) {
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
