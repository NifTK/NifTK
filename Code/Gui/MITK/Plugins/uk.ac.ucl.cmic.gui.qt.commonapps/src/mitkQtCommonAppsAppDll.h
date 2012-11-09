/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-26 10:10:39 +0100 (Fri, 26 Aug 2011) $
 Revision          : $Revision: 7170 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKQTCOMMONAPPSAPPDLL_H_
#define MITKQTCOMMONAPPSAPPDLL_H_

//
// The following block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the org_mitk_gui_qt_application_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// MITK_QT_APP functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.
//
#if defined(_WIN32) && !defined(BERRY_STATIC)
  #if defined(uk_ac_ucl_cmic_gui_qt_commonapps_EXPORTS)
    #define CMIC_QT_COMMONAPPS __declspec(dllexport)
  #else
    #define CMIC_QT_COMMONAPPS __declspec(dllimport)
  #endif
#endif

#if !defined(CMIC_QT_COMMONAPPS)
  #define CMIC_QT_COMMONAPPS
#endif

#endif /*MITKQTCOMMONAPPSAPPDLL_H_*/
