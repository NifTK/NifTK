/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkClickableLabel_p_h
#define niftkClickableLabel_p_h

#include <QLabel>


namespace niftk
{

/// \class ClickableLabel
/// \brief Custom QLabel that sends signals about mouse click events.
class ClickableLabel : public QLabel
{
  Q_OBJECT

public:

  /// \brief Constructs the ClickableLabel object.
  explicit ClickableLabel(QWidget *parent = 0);

  /// \brief Destructs the ClickableLabel object.
  virtual ~ClickableLabel();

signals:

  void clicked();

protected:

  void mousePressEvent(QMouseEvent* event) override;

};

}

#endif
