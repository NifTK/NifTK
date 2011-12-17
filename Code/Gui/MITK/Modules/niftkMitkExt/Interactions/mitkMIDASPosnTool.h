/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKMIDASPOSNTOOL_H
#define MITKMIDASPOSNTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkMIDASTool.h"

namespace mitk {

  /**
   * \class MIDASPosnTool
   * \brief Dummy class, as the MIDAS posn tool, just enables you to change the
   * position of the slices in 2 or 3 windows, which is the default behaviour of
   * the MITK ortho-viewer anyway.
   */
  class NIFTKMITKEXT_EXPORT MIDASPosnTool : public MIDASTool {

  public:
    mitkClassMacro(MIDASPosnTool, MIDASTool);
    itkNewMacro(MIDASPosnTool);

    virtual const char* GetName() const;
    virtual const char** GetXPM() const;

  protected:

    MIDASPosnTool(); // purposefully hidden
    virtual ~MIDASPosnTool(); // purposefully hidden

  private:

  };//class


}//namespace

#endif
