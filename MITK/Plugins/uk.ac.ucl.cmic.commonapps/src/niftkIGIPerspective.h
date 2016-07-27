/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIPerspective_h
#define niftkIGIPerspective_h

#include <uk_ac_ucl_cmic_commonapps_Export.h>
#include <berryIPerspectiveFactory.h>


namespace niftk
{

/**
 * \class IGIPerspective
 * \brief Perspective to arrange widgets as would be suitable for CMIC IGI applications.
 * \ingroup uk_ac_ucl_cmic_common
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class COMMONAPPS_EXPORT IGIPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)

public:

  IGIPerspective();
  IGIPerspective(const IGIPerspective& other);

  void CreateInitialLayout(berry::IPageLayout::Pointer layout) override;

};

}

#endif
