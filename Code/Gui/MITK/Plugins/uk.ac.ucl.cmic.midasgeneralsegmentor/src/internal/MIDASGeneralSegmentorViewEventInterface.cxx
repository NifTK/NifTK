/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MIDASGeneralSegmentorViewEventInterface.h"
#include "MIDASGeneralSegmentorView.h"

//-----------------------------------------------------------------------------
MIDASGeneralSegmentorViewEventInterface::MIDASGeneralSegmentorViewEventInterface()
: m_View(NULL)
{
}


//-----------------------------------------------------------------------------
MIDASGeneralSegmentorViewEventInterface::~MIDASGeneralSegmentorViewEventInterface()
{
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewEventInterface::SetMIDASGeneralSegmentorView( MIDASGeneralSegmentorView* view )
{
  m_View = view;
}


//-----------------------------------------------------------------------------
void MIDASGeneralSegmentorViewEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_View->ExecuteOperation(op);
}
