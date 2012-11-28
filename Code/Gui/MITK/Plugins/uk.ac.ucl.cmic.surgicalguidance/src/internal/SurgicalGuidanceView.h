/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#ifndef SurgicalGuidanceView_h
#define SurgicalGuidanceView_h

#include "QmitkBaseLegacyView.h"
#include "QmitkIGIDataSourceManager.h"

/**
 * \class SurgicalGuidanceView
 * \brief User interface to provide Image Guided Surgery functionality.
 * \ingroup uk_ac_ucl_cmic_surgicalguidance_internal
*/
class SurgicalGuidanceView : public QmitkBaseLegacyView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  SurgicalGuidanceView();
  virtual ~SurgicalGuidanceView();

  /// \brief Static view ID = uk.ac.ucl.cmic.surgicalguidance
  static const std::string VIEW_ID;

  /// \brief Returns the view ID.
  virtual std::string GetViewID() const;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

protected slots:

protected:

private slots:
  
private:

  QmitkIGIDataSourceManager::Pointer  m_DataSourceManager;
};

#endif // SurgicalGuidanceView_h
