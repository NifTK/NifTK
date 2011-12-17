/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-15 12:38:00 +0100 (Thu, 15 Sep 2011) $
 Revision          : $Revision: 7314 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "NonCopyableObject.h"
#include "Object.h"

#include <iostream>
using namespace std;

namespace niftk
{

NonCopyableObject::NonCopyableObject()
{
}

NonCopyableObject::~NonCopyableObject()
{
}

std::string NonCopyableObject::ToString() const
{
  // If this class has member variables, put them in this
  // string. At the moment it doesnt.
  return "NonCopyableObject[]";
}

}
