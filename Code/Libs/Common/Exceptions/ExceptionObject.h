/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-03-22 12:17:09 +0000 (Tue, 22 Mar 2011) $
 Revision          : $Rev: 5656 $
 Last modified by  : $LastChangedBy: sj $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __NIFTK_EXCEPTIONOBJECT_H
#define __NIFTK_EXCEPTIONOBJECT_H
#include <stdexcept>
#include <string>

namespace niftk
{
  /**
   * \brief Basic NIFTK exception class
   */
  class ExceptionObject : public std::runtime_error
  {
  public:
    ExceptionObject(const std::string &what) : std::runtime_error(what) {}
  };

} // end namespace

#endif
