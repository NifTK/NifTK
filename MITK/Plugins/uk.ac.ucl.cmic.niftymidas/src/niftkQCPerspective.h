/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQCPerspective_h
#define niftkQCPerspective_h

#include <uk_ac_ucl_cmic_niftymidas_Export.h>
#include <berryIPerspectiveFactory.h>

namespace niftk
{

/// \class QCPerspective
/// \brief Perspective for doing scan quality control (QC).
///
/// Note: We have to load at least one view component, to get an editor created.
class NIFTYMIDAS_EXPORT QCPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)

public:

  QCPerspective();
  QCPerspective(const QCPerspective& other);

  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

}

#endif
