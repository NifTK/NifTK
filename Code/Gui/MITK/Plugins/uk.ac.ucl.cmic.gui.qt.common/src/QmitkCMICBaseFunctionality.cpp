/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-09 09:28:51 +0100 (Fri, 09 Sep 2011) $
 Revision          : $Revision: 7274 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkCMICBaseFunctionality.h"
#include "QmitkStdMultiWidget.h"

QmitkCMICBaseFunctionality::QmitkCMICBaseFunctionality()
: m_MultiWidget(NULL)
, m_RenderingManagerObserverTag(0)
, m_SlicesRotationObserverTag1(0)
, m_SlicesRotationObserverTag2(0)
{
}

QmitkCMICBaseFunctionality::~QmitkCMICBaseFunctionality()
{
}

QmitkCMICBaseFunctionality::QmitkCMICBaseFunctionality(const QmitkCMICBaseFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

void QmitkCMICBaseFunctionality::Activated()
{
  itk::ReceptorMemberCommand<QmitkCMICBaseFunctionality>::Pointer command1 = itk::ReceptorMemberCommand<QmitkCMICBaseFunctionality>::New();
  command1->SetCallbackFunction( this, &QmitkCMICBaseFunctionality::RenderingManagerReinitialized );
  m_RenderingManagerObserverTag = mitk::RenderingManager::GetInstance()->AddObserver( mitk::RenderingManagerViewsInitializedEvent(), command1 );
}


void QmitkCMICBaseFunctionality::Deactivated()
{
  mitk::RenderingManager::GetInstance()->RemoveObserver( m_RenderingManagerObserverTag );
}

void QmitkCMICBaseFunctionality::StdMultiWidgetNotAvailable()
{
  SetMultiWidget(NULL);
}

void QmitkCMICBaseFunctionality::StdMultiWidgetClosed( QmitkStdMultiWidget& /*stdMultiWidget*/ )
{
  SetMultiWidget(NULL);
}

void QmitkCMICBaseFunctionality::StdMultiWidgetAvailable( QmitkStdMultiWidget& stdMultiWidget )
{
  SetMultiWidget(&stdMultiWidget);
}

void QmitkCMICBaseFunctionality::SetMultiWidget(QmitkStdMultiWidget* multiWidget)
{
  if (m_MultiWidget)
  {
    mitk::SlicesCoordinator* coordinator = m_MultiWidget->GetSlicesRotator();
    if (coordinator)
    {
      coordinator->RemoveObserver( m_SlicesRotationObserverTag1 );
    }
    coordinator = m_MultiWidget->GetSlicesSwiveller();
    if (coordinator)
    {
      coordinator->RemoveObserver( m_SlicesRotationObserverTag2 );
    }
  }

  m_MultiWidget = multiWidget;

  if (m_MultiWidget)
  {
    m_MultiWidget->DisableDepartmentLogo();

    mitk::SlicesCoordinator* coordinator = m_MultiWidget->GetSlicesRotator();
    if (coordinator)
    {
      itk::ReceptorMemberCommand<QmitkCMICBaseFunctionality>::Pointer command2 = itk::ReceptorMemberCommand<QmitkCMICBaseFunctionality>::New();
      command2->SetCallbackFunction( this, &QmitkCMICBaseFunctionality::SliceRotation );
      m_SlicesRotationObserverTag1 = coordinator->AddObserver( mitk::SliceRotationEvent(), command2 );
    }

    coordinator = m_MultiWidget->GetSlicesSwiveller();
    if (coordinator)
    {
      itk::ReceptorMemberCommand<QmitkCMICBaseFunctionality>::Pointer command2 = itk::ReceptorMemberCommand<QmitkCMICBaseFunctionality>::New();
      command2->SetCallbackFunction( this, &QmitkCMICBaseFunctionality::SliceRotation );
      m_SlicesRotationObserverTag2 = coordinator->AddObserver( mitk::SliceRotationEvent(), command2 );
    }
  }
}

void QmitkCMICBaseFunctionality::RenderingManagerReinitialized(const itk::EventObject&)
{
  // Deliberately do nothing.  So, if subclasses need this, they should override this method.
}

void QmitkCMICBaseFunctionality::SliceRotation(const itk::EventObject&)
{
  // Deliberately do nothing.  So, if subclasses need this, they should override this method.
}
