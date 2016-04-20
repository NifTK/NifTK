/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkSerializerMacros_h
#define __niftkSerializerMacros_h

// Note:
// MITK_REGISTER_SERIALIZER assumes that the type is in the 'mitk' namespace.
// To overcome this assumption, we create an 'alias' to our class with
// the desired name.

#define NIFTK_REGISTER_SERIALIZER(SERIALIZER_CLASS_NAME)\
namespace mitk\
{\
class SERIALIZER_CLASS_NAME : public niftk::SERIALIZER_CLASS_NAME\
{\
};\
\
MITK_REGISTER_SERIALIZER(SERIALIZER_CLASS_NAME);\
}\

#endif
