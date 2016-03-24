/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorEventInterface.h"
#include "niftkGeneralSegmentorView.h"

//-----------------------------------------------------------------------------
niftkGeneralSegmentorEventInterface::niftkGeneralSegmentorEventInterface()
: m_View(NULL)
{
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorEventInterface::~niftkGeneralSegmentorEventInterface()
{
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorEventInterface::SetGeneralSegmentorView( niftkGeneralSegmentorView* view )
{
  m_View = view;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_View->ExecuteOperation(op);
}
