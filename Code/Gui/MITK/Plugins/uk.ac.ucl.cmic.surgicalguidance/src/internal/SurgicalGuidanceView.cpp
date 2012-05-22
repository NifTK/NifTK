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

// Qt
#include <QMessageBox>

// IGI stuff, OpenIGTLink and NiftyLink
#include "igtlStringMessage.h"
#include "OIGTLSocketObject.h"

const std::string SurgicalGuidanceView::VIEW_ID = "uk.ac.ucl.cmic.surgicalguidance";

SurgicalGuidanceView::SurgicalGuidanceView()
{
  // Matt: I'm just creating a NiftyLink / OpenIGTLink function to check that include paths, and library linkage works.
  igtl::StringMessage::Pointer myFirstMessage = igtl::StringMessage::New();
  OIGTLSocketObject socket;

}

SurgicalGuidanceView::~SurgicalGuidanceView()
{
}

void SurgicalGuidanceView::CreateQtPartControl( QWidget *parent )
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );

  // connect signals-slots etc.
}

void SurgicalGuidanceView::SetFocus()
{
}
