/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMeshingUnitTestHelpers_h
#define niftkMeshingUnitTestHelpers_h

#define niftkMeshingAssert(COND) \
     {\
      if(!(COND))\
      {\
         std::cerr << "Test failed " << __FILE__ << ":" << __LINE__;\
         std::cerr << " in function " << __FUNCTION__ << std::endl;\
         std::cerr << "Condition: " << #COND;\
         abort();\
      }\
   }

#endif /* MESHINGUNITTESTHELPERS_H_ */
