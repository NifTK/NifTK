/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkWheelEventEater_h
#define niftkWheelEventEater_h

#include "niftkCoreGuiExports.h"
#include <QWidget>
#include <QEvent>

namespace niftk
{

/**
* \class QmitkWheelEventEater
* \brief Qt event filter to eat wheel events.
*/
class NIFTKCOREGUI_EXPORT QmitkWheelEventEater : public QObject
{
  Q_OBJECT

public:
  QmitkWheelEventEater(QWidget* parent=NULL) : QObject(parent) { m_IsEating = true; }
  ~QmitkWheelEventEater() {}
  void SetIsEating(bool b) { m_IsEating = b; }
  bool GetIsEating() const { return m_IsEating; }
protected:
  virtual bool eventFilter(QObject *obj, QEvent *event) override
  {
    if (m_IsEating && event->type() == QEvent::Wheel)
    {
      return true;
    } else
    {
      // standard event processing
      return QObject::eventFilter(obj, event);
    }
  }
private:
  bool m_IsEating;
}; // end class

} // end namespace

#endif
