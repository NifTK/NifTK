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


const std::string niftkMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::niftkMorphologicalSegmentorView()
: niftkBaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::niftkMorphologicalSegmentorView(
    const niftkMorphologicalSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::~niftkMorphologicalSegmentorView()
{
}


//-----------------------------------------------------------------------------
std::string niftkMorphologicalSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController* niftkMorphologicalSegmentorView::CreateSegmentorController()
{
  return new niftkMorphologicalSegmentorController(this);
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::SetFocus()
{
}


//-----------------------------------------------------------------------------
QString niftkMorphologicalSegmentorView::GetPreferencesNodeName()
{
  return niftkMorphologicalSegmentorPreferencePage::PREFERENCES_NODE_NAME;
}
