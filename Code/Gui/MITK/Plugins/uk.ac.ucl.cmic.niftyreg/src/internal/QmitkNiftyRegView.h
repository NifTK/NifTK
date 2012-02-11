/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QmitkNiftyRegView_h
#define QmitkNiftyRegView_h

#include "ui_QmitkNiftyRegViewControls.h"
#include "berryISelectionListener.h"
#include "QmitkFunctionality.h"

/**
 * \class QmitkNiftyRegView
 * \brief GUI interface to enable the user to run the NiftyReg registration algorithm.
 * \ingroup uk.ac.ucl.cmic.niftyreg
*/
class QmitkNiftyRegView : public QmitkFunctionality
{  
  Q_OBJECT
  
  public:  

    static const std::string VIEW_ID;

    QmitkNiftyRegView();
    virtual ~QmitkNiftyRegView();
    virtual void CreateQtPartControl(QWidget *parent);

  protected slots:

  protected:

    Ui::QmitkNiftyRegViewControls m_Controls;

};

#endif // QmitkNiftyRegView_h

