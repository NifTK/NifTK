/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAtracsysManager.h"
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>
#include <ftkInterface.h>

namespace niftk
{

//-----------------------------------------------------------------------------
class AtracsysManagerPrivate
{
  Q_DECLARE_PUBLIC(AtracsysManager)
  AtracsysManager* const q_ptr;

public:

  AtracsysManagerPrivate(AtracsysManager* q);
  ~AtracsysManagerPrivate();

  ftkLibrary m_Lib;

};


//-----------------------------------------------------------------------------
AtracsysManagerPrivate::AtracsysManagerPrivate(AtracsysManager* q)
: q_ptr(q)
, m_Lib(nullptr)
{

  m_Lib = ftkInit();
  if ( m_Lib == nullptr )
  {
    mitkThrow() << "Cannot initialize Atracsys driver.";
  }
}


//-----------------------------------------------------------------------------
AtracsysManagerPrivate::~AtracsysManagerPrivate()
{
  Q_Q(AtracsysManager);

  if (m_Lib != nullptr)
  {
    ftkClose( &m_Lib );
  }
}


//-----------------------------------------------------------------------------
AtracsysManager::AtracsysManager()
: d_ptr(new AtracsysManagerPrivate(this))
{
  MITK_INFO << "Creating AtracsysManager";
}


//-----------------------------------------------------------------------------
AtracsysManager::~AtracsysManager()
{
}

} // end namespace
