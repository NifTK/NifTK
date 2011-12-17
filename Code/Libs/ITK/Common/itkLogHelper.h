/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __ITKLOGHELPER_H
#define __ITKLOGHELPER_H

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include <iostream>
#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkIndent.h"
#include "itkUCLMacro.h"


namespace niftk {

/**
 * \class itkLogHelper
 * \brief This is a class to help with a few logging functions.
 */
class NIFTKITK_WINEXPORT itkLogHelper : public itk::Object
{

  public:

	/**
     * House keeping for the object factory.
     */
    typedef itkLogHelper Self;
    typedef itk::Object Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

  /* Method for creation through the object factory. */
    itkNewMacro(Self);

    /**
    * Runtime information support.
    */
    itkTypeMacro(itkLogHelper, Object);

    void PrintSelf(std::ostream& os, itk::Indent indent) const;

     /**
     * Set the logging level at run-time.
     */
    static void SetLogLevel(int  logginglevel);

    /**
     * Simple way to output an information
     */
    void InfoMessage(const std::string& infoMessage);

    /**
      * Simple way to output a debug message
      */
    void DebugMessage(const std::string& debugMessage);

    /**
      * Simple way to output a warning message
      */
    void WarningMessage(const std::string& warningMessage);

    /**
     * Simple way to output an error message
     */
    void ErrorMessage(const std::string& errorMessage);

    /**
     * Simple way to output a exception message
     */
    void ExceptionMessage(const std::string& exceptionMessage);

    /**
     * Prints out command line header
     */
    static void PrintCommandLineHeader(std::ostream& stream);

    /** Helper method, just in case we want to debug a parameter array. */
    static std::string WriteParameterArray(const itk::Array<double>& array);

    /** Prints out a string representation of this object. */
    std::string ToString();

  protected:
    itkLogHelper();
    virtual ~itkLogHelper();

  private:
    itkLogHelper(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

};  //end of class itkLogHelper

} //end of namespace niftk

#endif
