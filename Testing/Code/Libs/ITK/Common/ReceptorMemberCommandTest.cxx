/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include "itkSmartPointer.h"
#include "itkCommand.h"
#include "itkImage.h"
#include "itkAddImageFilter.h"

namespace mitk
{
  class Manager : public itk::Object
  {
    public:
      typedef Manager        Self;
      typedef itk::Object    Superclass;
      typedef itk::SmartPointer<Self> Pointer;
      typedef itk::SmartPointer<const Self>  ConstPointer;
      itkTypeMacro(Manager, itk::Object)
      itkNewMacro(Self);
  };
}

typedef itk::Image<short, 2> ImageType;
typedef itk::AddImageFilter<ImageType, ImageType> FilterType;
typedef mitk::Manager ManagerType;

class Base {
  public:
    Base() { m_Count = 0; }
    int GetCount() const { return m_Count; }
    void IncrementCount()
    {
      m_Count++;
      std::cerr << "m_Count=" << m_Count << std::endl;
    }
  private:
    int m_Count;
};

class Foo : public Base {
  public:
    Foo() { m_Filter = NULL; }
    void SetFilter(FilterType::Pointer filter)
    {
      m_Filter=filter;

      itk::ReceptorMemberCommand<Foo>::Pointer command =
          itk::ReceptorMemberCommand<Foo>::New();
      command->SetCallbackFunction(this, &Foo::OnLevelWindowModified );
      m_Filter->AddObserver( itk::ModifiedEvent(), command);

    }
    void OnLevelWindowModified(const itk::EventObject& )
    {
      this->IncrementCount();
    }
  private:
    FilterType::Pointer m_Filter;
};

class Bar : public Base {
  public:
    Bar() { m_Manager = NULL; }
    void SetManager(ManagerType::Pointer manager)
    {
      m_Manager=manager;

      itk::ReceptorMemberCommand<Bar>::Pointer command =
          itk::ReceptorMemberCommand<Bar>::New();
      command->SetCallbackFunction(this, &Bar::OnLevelWindowModified );
      m_Manager->AddObserver( itk::ModifiedEvent(), command);

    }
    void OnLevelWindowModified(const itk::EventObject& )
    {
      this->IncrementCount();
    }
  private:
    ManagerType::Pointer m_Manager;
};

int ReceptorMemberCommandTest(int argc, char * argv[])
{
  FilterType::Pointer filter = FilterType::New();
  ManagerType::Pointer manager = ManagerType::New();

  Foo *myFoo = new Foo();
  myFoo->SetFilter(filter);

  Bar *myBar = new Bar();
  myBar->SetManager(manager);

  filter->Modified();
  int count = myFoo->GetCount();
  if (count != 1)
  {
    return EXIT_FAILURE;
  }

  manager->Modified();
  count = myBar->GetCount();
  if (count != 1)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

