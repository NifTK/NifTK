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
#ifndef __NIFTK_OBJECT_H
#define __NIFTK_OBJECT_H

#include <iostream>
using namespace std;

#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"

namespace niftk
{

/**
 * \class Object
 * \brief Base Class for NIFTK specific stuff.
 * 
 * 
 **/
class NIFTKCOMMON_WINEXPORT Object {

  public:
    
    /** These strings used in logging statements. */
    static const std::string CONSTRUCTED;
    static const std::string DESTROYING;
    static const std::string DESTROYED;
    static const std::string COPYCONSTRUCTED;
    static const std::string COPYASSIGNED;
  
    Object();
    virtual ~Object() = 0;
  
    /**
     * Supply a 1 line string describing the state of this object.
     **/
    std::string ToString() const;
};

}

#endif

