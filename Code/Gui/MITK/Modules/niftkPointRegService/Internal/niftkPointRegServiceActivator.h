/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointRegServiceActivator_h
#define niftkPointRegServiceActivator_h

#include "niftkPointRegServiceUsingSVD.h"
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <memory>

namespace niftk
{

/**
* @class PointRegServiceActivator
* @brief Activator to register instances of niftk::PointRegServiceI, currently only niftk::PointRegServiceUsingSVD.
*/
class PointRegServiceActivator : public us::ModuleActivator
{
public:

  PointRegServiceActivator();
  ~PointRegServiceActivator();

  void Load(us::ModuleContext* context);
  void Unload(us::ModuleContext*);

private:

  PointRegServiceActivator(const PointRegServiceActivator&); // deliberately not implemented
  PointRegServiceActivator& operator=(const PointRegServiceActivator&); // deliberately not implemented

  std::auto_ptr<niftk::PointRegServiceUsingSVD> m_PointRegServiceSVD;
};

} // end namespace

#endif
