/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkBaseGUI_h
#define __niftkBaseGUI_h

#include <niftkCoreGuiExports.h>

#include <QObject>

class QWidget;

namespace niftk
{

/// \class BaseGUI
/// \brief Base class for GUI controls on BlueBerry views.
class NIFTKCOREGUI_EXPORT BaseGUI : public QObject
{
  Q_OBJECT

public:

  BaseGUI(QWidget* parent);
  virtual ~BaseGUI();

public:

  /// \brief Returns the parent widget.
  QWidget* GetParent() const;

};

}

#endif
