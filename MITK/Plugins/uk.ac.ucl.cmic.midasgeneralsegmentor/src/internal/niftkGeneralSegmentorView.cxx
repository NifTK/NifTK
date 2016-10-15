/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorView.h"

#include <niftkGeneralSegmentorController.h>

#include "niftkGeneralSegmentorPreferencePage.h"

namespace niftk
{

const QString GeneralSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor";


//-----------------------------------------------------------------------------
GeneralSegmentorView::GeneralSegmentorView()
  : BaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
GeneralSegmentorView::GeneralSegmentorView(
    const GeneralSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
GeneralSegmentorView::~GeneralSegmentorView()
{
}


//-----------------------------------------------------------------------------
BaseSegmentorController* GeneralSegmentorView::CreateSegmentorController()
{
  m_GeneralSegmentorController = new GeneralSegmentorController(this);
  return m_GeneralSegmentorController;
}


//-----------------------------------------------------------------------------
QString GeneralSegmentorView::GetPreferencesNodeName()
{
  return this->GetViewID();
}


//-----------------------------------------------------------------------------
QString GeneralSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void GeneralSegmentorView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}

}
