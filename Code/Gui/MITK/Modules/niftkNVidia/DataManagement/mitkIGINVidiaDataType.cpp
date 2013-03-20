/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkIGINVidiaDataType.h"

namespace mitk
{

//-----------------------------------------------------------------------------
IGINVidiaDataType::IGINVidiaDataType()
  : magic_cookie(0), sequence_number(0), gpu_arrival_time(0)
{
}

//-----------------------------------------------------------------------------
IGINVidiaDataType::~IGINVidiaDataType()
{
}

void IGINVidiaDataType::set_values(unsigned int cookie, unsigned int sn, unsigned __int64 gputime)
{
  magic_cookie = cookie;
  sequence_number = sn;
  gpu_arrival_time = gputime;
}

} // end namespace

