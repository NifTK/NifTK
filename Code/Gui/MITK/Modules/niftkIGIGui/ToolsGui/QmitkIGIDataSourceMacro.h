/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#define NIFTK_IGISOURCE_MACRO(EXPORT_SPEC, CLASS_NAME, DESCRIPTION) \
\
class EXPORT_SPEC CLASS_NAME ## Factory : public ::itk::ObjectFactoryBase \
{ \
  public: \
    \
    /* ITK typedefs */ \
    typedef CLASS_NAME ## Factory   Self; \
    typedef itk::ObjectFactoryBase  Superclass; \
    typedef itk::SmartPointer<Self>  Pointer; \
    typedef itk::SmartPointer<const Self>  ConstPointer; \
    \
    /* Methods from ObjectFactoryBase */ \
    virtual const char* GetITKSourceVersion() const \
    {\
      return ITK_SOURCE_VERSION; \
    }\
    \
    virtual const char* GetDescription() const \
    {\
      return DESCRIPTION; \
    }\
    \
    /* Method for class instantiation. */ \
    itkFactorylessNewMacro(Self); \
    \
    /* Run-time type information (and related methods). */ \
    itkTypeMacro(CLASS_NAME ## Factory, itkObjectFactoryBase); \
    \
    /* Register one factory of this type  */ \
    static void RegisterOneFactory() \
    {\
      CLASS_NAME ## Factory::Pointer CLASS_NAME ## Factory = CLASS_NAME ## Factory::New(); \
      itk::ObjectFactoryBase::RegisterFactory(CLASS_NAME ## Factory); \
    }\
    \
  protected: \
    \
    CLASS_NAME ## Factory() \
    {\
      itk::ObjectFactoryBase::RegisterOverride("mitk::IGIDataSource", \
                              #CLASS_NAME, \
                              DESCRIPTION, \
                              1, \
                              itk::CreateObjectFunction<CLASS_NAME>::New()); \
    }\
    \
    ~CLASS_NAME ## Factory() \
    { \
    } \
    \
  private: \
    \
    CLASS_NAME ## Factory(const Self&); /* purposely not implemented */ \
    void operator=(const Self&);    /* purposely not implemented */ \
    \
}; \
\
class CLASS_NAME ## RegistrationMethod \
  {\
    public: \
    \
    CLASS_NAME ## RegistrationMethod() \
    {\
      /*MITK_INFO("tools") << "Registered " #CLASS_NAME; */ \
      m_Factory = CLASS_NAME ## Factory::New(); \
      itk::ObjectFactoryBase::RegisterFactory( m_Factory ); \
    }\
    \
    ~CLASS_NAME ## RegistrationMethod() \
    {\
      /*MITK_INFO("tools") << "UnRegistered " #CLASS_NAME; */ \
      itk::ObjectFactoryBase::UnRegisterFactory( m_Factory ); \
    }\
    \
    private: \
    \
    CLASS_NAME ## Factory::Pointer m_Factory; \
  };\
\
static CLASS_NAME ## RegistrationMethod somestaticinitializer_ ## CLASS_NAME ;

#define NIFTK_IGISOURCE_GUI_MACRO(EXPORT_SPEC, CLASS_NAME, DESCRIPTION) \
\
class EXPORT_SPEC CLASS_NAME ## Factory : public ::itk::ObjectFactoryBase \
{\
  public: \
    \
    /* ITK typedefs */ \
    typedef CLASS_NAME ## Factory   Self; \
    typedef itk::ObjectFactoryBase  Superclass; \
    typedef itk::SmartPointer<Self>  Pointer; \
    typedef itk::SmartPointer<const Self>  ConstPointer; \
    \
    /* Methods from ObjectFactoryBase */ \
    virtual const char* GetITKSourceVersion() const \
    {\
      return ITK_SOURCE_VERSION; \
    }\
    \
    virtual const char* GetDescription() const \
    {\
      return DESCRIPTION; \
    }\
    \
    /* Method for class instantiation. */ \
    itkFactorylessNewMacro(Self); \
    \
    /* Run-time type information (and related methods). */ \
    itkTypeMacro(CLASS_NAME ## Factory, itkObjectFactoryBase); \
    \
    /* Register one factory of this type  */ \
    static void RegisterOneFactory() \
    {\
      CLASS_NAME ## Factory::Pointer CLASS_NAME ## Factory = CLASS_NAME ## Factory::New(); \
      itk::ObjectFactoryBase::RegisterFactory(CLASS_NAME ## Factory); \
    }\
    \
  protected: \
    \
    CLASS_NAME ## Factory() \
    {\
      itk::ObjectFactoryBase::RegisterOverride(#CLASS_NAME, \
                              #CLASS_NAME, \
                              DESCRIPTION, \
                              1, \
                              itk::CreateObjectFunction<CLASS_NAME>::New()); \
    }\
    \
    ~CLASS_NAME ## Factory() \
    { \
    } \
    \
  private: \
    \
    CLASS_NAME ## Factory(const Self&); /* purposely not implemented */ \
    void operator=(const Self&);    /* purposely not implemented */ \
    \
}; \
\
class CLASS_NAME ## RegistrationMethod \
  {\
    public: \
    \
    CLASS_NAME ## RegistrationMethod() \
    { \
      /*MITK_INFO("tools") << "Registered " #CLASS_NAME; */ \
      m_Factory = CLASS_NAME ## Factory::New(); \
      itk::ObjectFactoryBase::RegisterFactory( m_Factory ); \
    } \
    \
    ~CLASS_NAME ## RegistrationMethod() \
     { \
       /*MITK_INFO("tools") << "UnRegistered " #CLASS_NAME; */ \
       itk::ObjectFactoryBase::UnRegisterFactory( m_Factory ); \
     } \
     \
     private: \
     \
     CLASS_NAME ## Factory::Pointer m_Factory; \
  }; \
\
static CLASS_NAME ## RegistrationMethod somestaticinitializer_ ## CLASS_NAME ;
