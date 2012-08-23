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

// Qmitk
#include "SurgicalGuidanceView.h"

const std::string SurgicalGuidanceView::VIEW_ID = "uk.ac.ucl.cmic.surgicalguidance";

//-----------------------------------------------------------------------------
SurgicalGuidanceView::SurgicalGuidanceView()
{
}


//-----------------------------------------------------------------------------
SurgicalGuidanceView::~SurgicalGuidanceView()
{
}


//-----------------------------------------------------------------------------
std::string SurgicalGuidanceView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceView::CreateQtPartControl( QWidget *parent )
{
  m_ToolManager = QmitkIGIToolManager::New();
  m_ToolManager->setupUi(parent);
  m_ToolManager->SetStdMultiWidget(this->GetActiveStdMultiWidget());
  m_ToolManager->SetDataStorage(this->GetDataStorage());
}


//-----------------------------------------------------------------------------
void SurgicalGuidanceView::SetFocus()
{
  m_ToolManager->setFocus();
}
