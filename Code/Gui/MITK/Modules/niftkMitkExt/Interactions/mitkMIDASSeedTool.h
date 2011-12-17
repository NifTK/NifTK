/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-19 22:31:43 +0000 (Sat, 19 Nov 2011) $
 Revision          : $Revision: 7815 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKMIDASSEEDTOOL_H
#define MITKMIDASSEEDTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkMIDASTool.h"
#include "mitkStateEvent.h"
#include "mitkMIDASPointSetInteractor.h"

namespace mitk {

  /**
   * \class MIDASSeedTool
   * \brief MIDAS seed tool for adding / removing / moving seeds.
   *
   * Interestingly, ANY of the MIDAS tools, PolyTool, DrawTool etc can add seeds.
   * but this is the only tool that can remove them.
   */
  class NIFTKMITKEXT_EXPORT MIDASSeedTool : public MIDASTool {

  public:

    mitkClassMacro(MIDASSeedTool, MIDASTool);
    itkNewMacro(MIDASSeedTool);

    virtual const char* GetName() const;
    virtual const char** GetXPM() const;

    // When called, we create and register an mitkPointSetInteractor.
    virtual void Activated();

    // When called, we unregister the mitkPointSetInteractor.
    virtual void Deactivated();

  protected:

    MIDASSeedTool();
    virtual ~MIDASSeedTool();

  private:

    mitk::MIDASPointSetInteractor::Pointer m_PointSetInteractor;

  };//class


}//namespace

#endif
