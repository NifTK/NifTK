/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkDistanceMeasurerGUI_h
#define niftkDistanceMeasurerGUI_h

#include <QWidget>
#include "ui_niftkDistanceMeasurerGUI.h"
#include <niftkBaseGUI.h>
#include <mitkDataStorage.h>

namespace niftk
{

/// \class DistanceMeasurerGUI
/// \brief Widgets for Distance Measurer plugin.
class DistanceMeasurerGUI : public BaseGUI,
                            private Ui::DistanceMeasurerGUI
{

  Q_OBJECT

public:

  DistanceMeasurerGUI(QWidget* parent);
  virtual ~DistanceMeasurerGUI();

  void SetDataStorage(mitk::DataStorage* storage);
  void Reset();

signals:

  void LeftImageSelectionChanged(const mitk::DataNode*);
  void LeftMaskSelectionChanged(const mitk::DataNode*);
  void RightImageSelectionChanged(const mitk::DataNode*);
  void RightMaskSelectionChanged(const mitk::DataNode*);

private:

  DistanceMeasurerGUI(const DistanceMeasurerGUI&);  // Purposefully not implemented.
  void operator=(const DistanceMeasurerGUI&);  // Purposefully not implemented.
};

} // end namespace

#endif
