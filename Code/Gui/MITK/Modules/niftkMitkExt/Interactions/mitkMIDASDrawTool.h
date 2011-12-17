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
#ifndef MITKMIDASDRAWTOOL_H
#define MITKMIDASDRAWTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkMIDASContourTool.h"

namespace mitk {

  /**
   * \class MIDASDrawTool
   * \brief Tool to draw lines around voxel edges rather than through them.
   */
  class NIFTKMITKEXT_EXPORT MIDASDrawTool : public MIDASContourTool {

  public:

    mitkClassMacro(MIDASDrawTool, MIDASContourTool);
    itkNewMacro(MIDASDrawTool);

    virtual const char* GetName() const;
    virtual const char** GetXPM() const;

    virtual bool OnLeftMousePressed (Action* action, const StateEvent* stateEvent);
    virtual bool OnLeftMouseMoved   (Action* action, const StateEvent* stateEvent);
    virtual bool OnLeftMouseReleased(Action* action, const StateEvent* stateEvent);

    // Wipe's all the contours.
    virtual void Wipe();

  protected:

    MIDASDrawTool(); // purposely hidden
    virtual ~MIDASDrawTool(); // purposely hidden

  private:

    // Not thread safe
    mitk::Point3D m_MostRecentPointInMillimetres;

  };//class


}//namespace

#endif
