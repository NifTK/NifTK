/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkThumbnailInteractor_h
#define mitkThumbnailInteractor_h

#include <niftkThumbnailExports.h>

#include <mitkDisplayInteractor.h>

namespace mitk
{
class BaseRenderer;
class SliceNavigationController;

/**
 *\class ThumbnailInteractor
 *@brief Observer that manages the interaction with the thumbnail window.
 *
 * @ingroup Interaction
 **/
class niftkThumbnail_EXPORT ThumbnailInteractor: public DisplayInteractor
{
public:
  mitkClassMacro(ThumbnailInteractor, DisplayInteractor)

  // Thumbnail window customisation: renderer passed as argument
  mitkNewMacro1Param(Self, mitk::BaseRenderer*);

  /**
   * By this function the Observer gets notifier about new events.
   * Here it is adapted to pass the events to the state machine in order to use
   * its infrastructure.
   * It also checks if event is to be accepted when i already has been processed by a DataInteractor.
   */
  virtual void Notify(InteractionEvent* interactionEvent, bool isHandled);

protected:
  ThumbnailInteractor(mitk::BaseRenderer* renderer);
  virtual ~ThumbnailInteractor();

  virtual void ConnectActionsAndFunctions();

  virtual bool Init(StateMachineAction* action, InteractionEvent* event);
  virtual bool InitZoom(StateMachineAction* action, InteractionEvent* event);
  virtual bool Move(StateMachineAction* action, InteractionEvent* event);
  virtual bool Zoom(StateMachineAction* action, InteractionEvent* event);

private:
  /**
   * Renderer of the thumbnail render window that this display interactor belongs to.
   */
  mitk::BaseRenderer* m_Renderer;

  /**
   * Slice navigation controller of the thumbnail render window that this display interactor belongs to.
   */
  mitk::SliceNavigationController* m_SliceNavigationController;
};
}
#endif
