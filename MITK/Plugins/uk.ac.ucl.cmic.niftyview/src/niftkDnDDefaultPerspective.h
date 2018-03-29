/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDnDDefaultPerspective_h
#define niftkDnDDefaultPerspective_h

#include <uk_ac_ucl_cmic_niftyview_Export.h>
#include <berryIPerspectiveFactory.h>

namespace niftk
{

/// \class DnDDefaultPerspective
/// \brief Default perspective for Drag and Drop display in NiftyView.
///
/// Note: We have to load at least one view component, to get an editor created.
class NIFTYVIEW_EXPORT DnDDefaultPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)

public:

  DnDDefaultPerspective();
  DnDDefaultPerspective(const DnDDefaultPerspective& other);

  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

}

#endif
