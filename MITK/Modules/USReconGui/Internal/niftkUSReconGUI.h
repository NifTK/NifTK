/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkUSReconGUI_h
#define niftkUSReconGUI_h

#include <QWidget>
#include "ui_niftkUSReconGUI.h"
#include <niftkBaseGUI.h>
#include <mitkDataStorage.h>

namespace niftk
{

/// \class USReconGUI
/// \brief Widgets for Ultrasound Reconstruction plugin.
class USReconGUI : public BaseGUI,
                   private Ui::USReconGUI
{

  Q_OBJECT

public:

  USReconGUI(QWidget* parent);
  virtual ~USReconGUI();

  void SetDataStorage(mitk::DataStorage* storage);
  void SetEnableButtons(bool isEnabled);
  mitk::DataNode::Pointer GetImageNode() const;
  mitk::DataNode::Pointer GetTrackingNode() const;

signals:

  void OnImageSelectionChanged(const mitk::DataNode*);
  void OnTrackingSelectionChanged(const mitk::DataNode*);
  void OnGrabPressed();
  void OnReconstructPressed();
  void OnClearDataPressed();

private:

  USReconGUI(const USReconGUI&);  // Purposefully not implemented.
  void operator=(const USReconGUI&);  // Purposefully not implemented.
};

} // end namespace

#endif
