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
   * Interestingly, ANY of mitk::MIDASPolyTool, mitk::MIDASDrawTool or mitk::MIDASSeedTool can add seeds.
   * but only mitk::MIDASSeedTool can remove them.
   *
   * Provides
   * <pre>
   * 1. Right mouse button = place seed
   * 2. Left mouse button = select seed
   * 3. Move with left mouse button down = move selected seed
   * 4. Middle mouse button = select a seed if it is within a given distance and remove it.
   * </pre>
   * and includes Undo/Redo functionality. Given the above list, to remove seeds most people
   * hold the middle mouse button down, and drag it around, sucking up the seed points like a hoover.
   */
  class NIFTKMITKEXT_EXPORT MIDASSeedTool : public MIDASTool {

  public:

    mitkClassMacro(MIDASSeedTool, MIDASTool);
    itkNewMacro(MIDASSeedTool);

    /// \see mitk::Tool::GetName()
    virtual const char* GetName() const;

    /// \see mitk::Tool::GetXPM()
    virtual const char** GetXPM() const;

    /// \brief When called, we create and register an mitkPointSetInteractor.
    virtual void Activated();

    /// \brief When called, we unregister the mitkPointSetInteractor.
    virtual void Deactivated();

    /// \see mitk::StateMachine::CanHandleEvent
    float CanHandleEvent(const StateEvent *) const;

  protected:

    MIDASSeedTool();
    virtual ~MIDASSeedTool();

  private:

    mitk::MIDASPointSetInteractor::Pointer m_PointSetInteractor;

  };//class


}//namespace

#endif
