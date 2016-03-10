/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkNiftySegView_h
#define QmitkNiftySegView_h

#include "ui_QmitkNiftySegViewControls.h"
#include <berryISelectionListener.h>
#include <QmitkAbstractView.h>

/**
 * \class QmitkNiftySegView
 * \brief GUI interface to enable the user to run the NiftySeg segmentation algorithm.
*/
class QmitkNiftySegView : public QmitkAbstractView
{  
  Q_OBJECT
  
  public:  

    static const std::string VIEW_ID;

    QmitkNiftySegView();
    virtual ~QmitkNiftySegView();

  protected:

    /// \brief Called by framework, this method creates all the controls for this view
    virtual void CreateQtPartControl(QWidget *parent);

    /// \brief Called by framework, sets the focus on a specific widget.
    virtual void SetFocus();

  protected slots:

    void OnClickedEMInitialisationRadioButtons(bool bClicked);

  protected:

    Ui::QmitkNiftySegViewControls m_Controls;

};

#endif // QmitkNiftySegView_h

