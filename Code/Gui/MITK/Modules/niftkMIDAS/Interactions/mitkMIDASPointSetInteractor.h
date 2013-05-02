/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_MIDASPointSetInteractor_h
#define mitk_MIDASPointSetInteractor_h

#include "niftkMIDASExports.h"
#include <mitkPointSetInteractor.h>

namespace mitk
{
  /**
   * \class mitkMIDASPointSetInteractor
   * \brief Derived from mitkPointSetInteractor so we can handle the mouse move event.
   * \ingroup Interaction
   */
  class NIFTKMIDAS_EXPORT MIDASPointSetInteractor : public PointSetInteractor
  {
  public:
    mitkClassMacro(MIDASPointSetInteractor, PointSetInteractor);
    mitkNewMacro3Param(Self, const char*, DataNode*, int);
    mitkNewMacro2Param(Self, const char*, DataNode*);

    /**
     * \brief overriden the base class function, to enable mouse move events.
     */
    virtual float CanHandleEvent(StateEvent const* stateEvent) const;

  protected:
    /**
     * \brief Constructor with Param n for limited Set of Points
     *
     * If no n is set, then the number of points is unlimited
     * n=0 is not supported. In this case, n is set to 1.
     */
    MIDASPointSetInteractor(const char * type, DataNode* dataNode, int n = -1);

    /**
     * \brief Default Destructor
     **/
    virtual ~MIDASPointSetInteractor();

  private:

  };
}
#endif /* MITKMIDASPOINTSETINTERACTOR_H */
