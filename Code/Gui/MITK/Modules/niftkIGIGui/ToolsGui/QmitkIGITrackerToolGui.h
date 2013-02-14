/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGITRACKERTOOLGUI_H
#define QMITKIGITRACKERTOOLGUI_H

#include "niftkIGIGuiExports.h"
#include "QmitkIGINiftyLinkDataSourceGui.h"
#include "ui_QmitkIGITrackerToolGui.h"

class ClientDescriptorXMLBuilder;
class QmitkIGITrackerTool;
class QmitkFiducialRegistrationWidgetDialog;

/**
 * \class QmitkIGITrackerToolGui
 * \brief Base class for IGI Tracker Tool GUIs.
 */
class NIFTKIGIGUI_EXPORT QmitkIGITrackerToolGui : public QmitkIGINiftyLinkDataSourceGui, public Ui_QmitkIGITrackerToolGui
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGITrackerToolGui, QmitkIGINiftyLinkDataSourceGui);
  itkNewMacro(QmitkIGITrackerToolGui);

protected:

  QmitkIGITrackerToolGui(); // Purposefully hidden.
  virtual ~QmitkIGITrackerToolGui(); // Purposefully hidden.

  QmitkIGITrackerToolGui(const QmitkIGITrackerToolGui&); // Purposefully not implemented.
  QmitkIGITrackerToolGui& operator=(const QmitkIGITrackerToolGui&); // Purposefully not implemented.

  /**
   * \brief Initializes this widget, calling Ui_QmitkIGITrackerToolGui::setupUi(parent),
   * and any other stuff as necessary.
   */
  virtual void Initialize(QWidget *parent, ClientDescriptorXMLBuilder* config);

private slots:

  void OnStartTrackingClicked(void);
  void OnGetCurrentPosition(void);
  void OnFiducialRegistrationClicked(void);
  void OnManageToolConnection(void);
  void OnAssocClicked(void);
  void OnDisassocClicked(void);
  void OnCameraLinkClicked(void);
  void OnLHCRHCClicked(void);
  void OnFidTrackClicked(void);
  void OnApplyFidClicked(void);
  void OnStatusUpdate(QString message);
  void OnRegisterFiducials();
  void OnGetTipPosition();
  void OnSetUpFinePositioning();

private:

  QmitkIGITrackerTool* GetQmitkIGITrackerTool() const;

  QmitkFiducialRegistrationWidgetDialog *m_FiducialRegWidgetDialog;
}; // end class

#endif

