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

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "SurgicalGuidanceView.h"
#include "QmitkStdMultiWidget.h"

// Qt
#include <QMessageBox>


const std::string SurgicalGuidanceView::VIEW_ID = "uk.ac.ucl.cmic.surgicalguidance";

SurgicalGuidanceView::SurgicalGuidanceView()
: m_Parent(NULL)
{
}

SurgicalGuidanceView::~SurgicalGuidanceView()
{
}

void SurgicalGuidanceView::CreateQtPartControl( QWidget *parent )
{
  m_Parent = parent;

  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );

  // connect signals-slots etc.
}
