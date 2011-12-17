/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-04-08 17:35:54 +0100 (Fri, 08 Apr 2011) $
 Revision          : $LastChangedRevision: 5828 $
 Last modified by  : $LastChangedBy: sj $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MESHINGUNITTESTHELPERS_H_
#define MESHINGUNITTESTHELPERS_H_

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
