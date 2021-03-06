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

#include <niftkBaseView.h>
#include <service/event/ctkEvent.h>
#include <QmitkRMSErrorWidget.h>

/**
 * \class RMSErrorView
 * \brief User interface to simply setup the QmitkRMSErrorWidget.
 * \ingroup uk_ac_ucl_cmic_igirmserror_internal
*/
class RMSErrorView : public niftk::BaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  RMSErrorView();
  virtual ~RMSErrorView();

  /**
   * \brief Each View for a plugin has its own globally unique ID, this one is
   * "uk.ac.ucl.cmic.igirmserror" and the .cxx file and plugin.xml should match.
   */
  static const QString VIEW_ID;

protected:

  /**
   *  \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent) override;

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus() override;

private slots:
  
  /**
   * \brief We listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

private:

  /**
   * \brief All the controls for the main view part.
   */
  QmitkRMSErrorWidget *m_Controls;
};

#endif // RMSErrorView_h
