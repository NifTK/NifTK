/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef RMSErrorView_h
#define RMSErrorView_h

#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_RMSErrorView.h"

/**
 * \class RMSErrorView
 * \brief User interface to simply setup the QmitkRMSErrorWidget.
 * \ingroup uk_ac_ucl_cmic_igirmserror_internal
*/
class RMSErrorView : public QmitkBaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  RMSErrorView();
  virtual ~RMSErrorView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igirmserror
   */
  static const std::string VIEW_ID;

  /**
   * \brief Returns the view ID.
   */
  virtual std::string GetViewID() const;

protected:

  /**
   *  \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent);

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus();

private slots:
  
  /**
   * \brief We listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

private:

  /**
   * \brief All the controls for the main view part.
   */
  Ui::RMSErrorView *m_Controls;
};

#endif // RMSErrorView_h
