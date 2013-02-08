/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef SnapshotView_h
#define SnapshotView_h

#include "QmitkAbstractView.h"
#include "ui_SnapshotViewControls.h"


/**
 * \class SnapshotView
 * \brief Simple user interface to provide screenshots of the current editor window.
 * \ingroup uk_ac_ucl_cmic_snapshot_internal
*/
class SnapshotView : public QmitkAbstractView
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
public:

  SnapshotView();
  virtual ~SnapshotView();

  /// \brief Static view ID = uk.ac.ucl.cmic.snapshot
  static const std::string VIEW_ID;

  /// \brief Returns the view ID.
  virtual std::string GetViewID() const;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

protected slots:
  
  virtual void OnTakeSnapshotButtonPressed();

protected:
  Ui::SnapshotViewControls *m_Controls;

private:

  QWidget* m_Parent;
};

#endif // SnapshotView_h

