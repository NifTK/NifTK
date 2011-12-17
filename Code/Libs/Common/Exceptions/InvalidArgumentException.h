/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-04-04 14:36:33 +0100 (Mon, 04 Apr 2011) $
 Revision          : $LastChangedRevision: 5755 $
 Last modified by  : $LastChangedByAuthor$

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef INVALIDARGUMENTEXCEPTION_H_
#define INVALIDARGUMENTEXCEPTION_H_

#include "ExceptionObject.h"

namespace niftk
{
  /**
   * \brief Exceptions for signalling invalid input.
   */
  class InvalidArgumentException : public ExceptionObject
  {
  public:
    InvalidArgumentException(const std::string &what) : ExceptionObject(what) {}
  };

} // end namespace


#endif /* INVALIDARGUMENTEXCEPTION_H_ */
