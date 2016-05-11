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


const std::string niftkGeneralSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasgeneralsegmentor";


//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::niftkGeneralSegmentorView()
  : niftkBaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::niftkGeneralSegmentorView(
    const niftkGeneralSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
niftkGeneralSegmentorView::~niftkGeneralSegmentorView()
{
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController* niftkGeneralSegmentorView::CreateSegmentorController()
{
  m_GeneralSegmentorController = new niftkGeneralSegmentorController(this);
  return m_GeneralSegmentorController;
}


//-----------------------------------------------------------------------------
QString niftkGeneralSegmentorView::GetPreferencesNodeName()
{
  return niftkGeneralSegmentorPreferencePage::PREFERENCES_NODE_NAME;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::Visible()
{
  niftkBaseSegmentorView::Visible();

  m_GeneralSegmentorController->OnViewGetsVisible();
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::Hidden()
{
  niftkBaseSegmentorView::Hidden();

  m_GeneralSegmentorController->OnViewGetsHidden();
}


//-----------------------------------------------------------------------------
std::string niftkGeneralSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void niftkGeneralSegmentorView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}
