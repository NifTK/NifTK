/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMinimalPerspective_h
#define niftkMinimalPerspective_h

#include <uk_ac_ucl_cmic_commonapps_Export.h>
#include <berryIPerspectiveFactory.h>


namespace niftk
{

/**
 * \class MinimalPerspective
 * \brief Default Perspective, called 'Minimal' to discourage incrementally adding to it.
 * \ingroup uk_ac_ucl_cmic_common
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class COMMONAPPS_EXPORT MinimalPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)

public:

  MinimalPerspective();
  MinimalPerspective(const MinimalPerspective& other);

  void CreateInitialLayout(berry::IPageLayout::Pointer layout) override;

};

}

#endif
