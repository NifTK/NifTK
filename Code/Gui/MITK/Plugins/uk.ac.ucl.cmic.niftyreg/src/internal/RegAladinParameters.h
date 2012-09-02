/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author:  $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef RegAladinParameters_h
#define RegAladinParameters_h

#include <QString>

#include "NiftyRegCommon.h"


/**
 * \class RegAladinParameters
 * \brief Class to store and initialise the parameters of the affine Aladin registration.
 * \ingroup uk.ac.ucl.cmic.niftyreg
*/


class RegAladinParameters
{  

  public:  

    RegAladinParameters();
    virtual ~RegAladinParameters();

    /// \brief Set the default parameters
    void SetDefaultParameters();

    /// \brief Print the object
    void PrintSelf( std::ostream& os );

    /// Assignment operator
    RegAladinParameters &operator=(const RegAladinParameters &p);

    QString referenceImageName; // -ref
    QString referenceImagePath; // -ref

    QString floatingImageName; // -flo
    QString floatingImagePath; // -flo

    QString referenceMaskName; // -rmask
    QString referenceMaskPath; // -rmask

    bool outputResultFlag;

    QString outputResultName; // -res
    QString outputResultPath; // -res

    bool outputAffineFlag;
    QString outputAffineName; // -aff

    // Aladin - Initialisation

    bool alignCenterFlag;        // -nac

    // Aladin - Method

    AffineRegistrationType regnType; // -rigOnly, -affDirect

    int maxiterationNumber;		// -maxit

    int block_percent_to_use; // -%v
    int inlier_lts;		// -%i

    // Aladin - Advanced

    InterpolationType interpolation;

};

#endif // RegAladinParameters_h

