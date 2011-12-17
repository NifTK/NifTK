/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-29 23:41:22 +0100 (Fri, 29 Jul 2011) $
 Revision          : $Revision: 6892 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASPOINTSETINTERACTOR_H
#define MITKMIDASPOINTSETINTERACTOR_H

#include "niftkMitkExtExports.h"
#include "mitkPointSetInteractor.h"

namespace mitk
{
  /**
   * \class mitkMIDASPointSetInteractor
   * \brief Derived from mitkPointSetInteractor so we can handle the mouse move event.
   * \ingroup Interaction
   */
  class NIFTKMITKEXT_EXPORT MIDASPointSetInteractor : public PointSetInteractor
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
