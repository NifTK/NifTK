/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkInteractionEventObserverMutex.h"

#include <usModuleContext.h>
#include <usServiceReference.h>

#include <mitkInteractionEventObserver.h>

namespace niftk
{

//-----------------------------------------------------------------------------
InteractionEventObserverMutex::InteractionEventObserverMutex()
: m_GuardedObserver(nullptr)
{
}


//-----------------------------------------------------------------------------
InteractionEventObserverMutex::~InteractionEventObserverMutex()
{
}


//-----------------------------------------------------------------------------
InteractionEventObserverMutex* InteractionEventObserverMutex::GetInstance()
{
  static InteractionEventObserverMutex* instance = nullptr;
  if (!instance)
  {
    instance = new InteractionEventObserverMutex();
  }
  return instance;
}


//-----------------------------------------------------------------------------
void InteractionEventObserverMutex::Lock(mitk::InteractionEventObserver* guardedObserver)
{
  assert(!m_GuardedObserver);

  m_GuardedObserver = guardedObserver;

  this->DeactivateOtherInteractionEventObservers();
}


//-----------------------------------------------------------------------------
void InteractionEventObserverMutex::Unlock(mitk::InteractionEventObserver* guardedObserver)
{
  assert(guardedObserver == m_GuardedObserver);

  this->ReactivateOtherInteractionEventObservers();

  m_GuardedObserver = nullptr;
}


//-----------------------------------------------------------------------------
void InteractionEventObserverMutex::DeactivateOtherInteractionEventObservers()
{
  us::ModuleContext* moduleContext = us::GetModuleContext();
  std::vector<us::ServiceReference<mitk::InteractionEventObserver>> interactionEventObserverRefs =
      moduleContext->GetServiceReferences<mitk::InteractionEventObserver>();

  m_StateOfOtherInteractionEventObservers.clear();

  for (auto interactionEventObserverRef: interactionEventObserverRefs)
  {
    mitk::InteractionEventObserver* interactionEventObserver =
        moduleContext->GetService<mitk::InteractionEventObserver>(interactionEventObserverRef);

    if (interactionEventObserver == m_GuardedObserver)
    {
      continue;
    }

    m_StateOfOtherInteractionEventObservers[interactionEventObserver] = interactionEventObserver->IsEnabled();
    interactionEventObserver->Disable();
  }
}


//-----------------------------------------------------------------------------
void InteractionEventObserverMutex::ReactivateOtherInteractionEventObservers()
{
  us::ModuleContext* moduleContext = us::GetModuleContext();
  std::vector<us::ServiceReference<mitk::InteractionEventObserver>> interactionEventObserverRefs =
      moduleContext->GetServiceReferences<mitk::InteractionEventObserver>();

  for (auto interactionEventObserverRef: interactionEventObserverRefs)
  {
    mitk::InteractionEventObserver* interactionEventObserver =
        moduleContext->GetService<mitk::InteractionEventObserver>(interactionEventObserverRef);

    if (interactionEventObserver == m_GuardedObserver)
    {
      continue;
    }

    auto otherObserverIt = m_StateOfOtherInteractionEventObservers.find(interactionEventObserver);
    assert(otherObserverIt != m_StateOfOtherInteractionEventObservers.end());

    bool wasEnabled = otherObserverIt->second;

    if (wasEnabled)
    {
      interactionEventObserver->Enable();
    }
    else
    {
      interactionEventObserver->Disable();
    }
  }
}

}
