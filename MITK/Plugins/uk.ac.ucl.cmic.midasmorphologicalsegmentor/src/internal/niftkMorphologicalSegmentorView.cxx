/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorView.h"

#include "niftkMorphologicalSegmentorPreferencePage.h"

#include <niftkMorphologicalSegmentorController.h>

namespace niftk
{

const QString MorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";


//-----------------------------------------------------------------------------
MorphologicalSegmentorView::MorphologicalSegmentorView()
: BaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
MorphologicalSegmentorView::MorphologicalSegmentorView(
    const MorphologicalSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
MorphologicalSegmentorView::~MorphologicalSegmentorView()
{
}


//-----------------------------------------------------------------------------
QString MorphologicalSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
BaseSegmentorController* MorphologicalSegmentorView::CreateSegmentorController()
{
  return new MorphologicalSegmentorController(this);
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorView::SetFocus()
{
}


//-----------------------------------------------------------------------------
QString MorphologicalSegmentorView::GetPreferencesNodeName()
{
  return this->GetViewID();
}

}
