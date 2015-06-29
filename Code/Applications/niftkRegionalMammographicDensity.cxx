/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>

#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkDOMReader.h>


#include <itkRegionalMammographicDensity.h>

#include <niftkRegionalMammographicDensityCLP.h>

namespace fs = boost::filesystem;



// -----------------------------------------------------------------------------------
// Read the pectoralis XML file
// -----------------------------------------------------------------------------------

void ReadXMLFile( std::string fileInputXML,
                 itk::DOMNode::Pointer &dom )
{
  itk::DOMNodeXMLReader::Pointer domReader = itk::DOMNodeXMLReader::New();

  domReader->SetFileName( fileInputXML );

  try {
    std::cout << "Reading XML file: " << fileInputXML << std::endl;
    domReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed to read XML file: " << err << std::endl;
    exit( EXIT_FAILURE );
  }

  dom = domReader->GetOutput();
}


// -----------------------------------------------------------------------------------
// Get the text value of a node
// -----------------------------------------------------------------------------------

std::string GetText( itk::DOMNode::ChildrenListType::iterator &itNode, std::string id )
{
  std::string str;

  itk::DOMNode::Pointer child = (*itNode)->GetChildByID( id );

  if ( ! child )
  {
    std::cerr << "ERROR: Could not find child node: " << id << std::endl;
    exit( EXIT_FAILURE );
  }

  itk::DOMTextNode::Pointer textNode = child->GetTextChild();

  if ( ! textNode )
  {
    std::cerr << "ERROR: Could not find text node: " << id << std::endl;
    exit( EXIT_FAILURE );
  }

  str = textNode->GetText();


  if ( ! str.length() )
  {
    std::cerr << "ERROR: Empty text value for: " << id << std::endl;
    exit( EXIT_FAILURE );
  }

  return str;
}


// --------------------------------------------------------------------------
// WriteHeaderToCSVFile()
// --------------------------------------------------------------------------

void WriteToCSVFile( std::ofstream *foutOutputDensityCSV )
{
  //                                   123456789012345678901234567890

  *foutOutputDensityCSV
    << std::right << std::setw(10) << "Patient id" << ", "

    << std::right << std::setw(17) << "Diagnostic ID"   << ", "
    << std::right << std::setw(60) << "Diagnostic file" << ", "
    << std::right << std::setw(18) << "Diag threshold"  << ", "

    << std::right << std::setw(15) << "Pre-diagnostic or Control?"   << ", "

    << std::right << std::setw(17) << "Pre-diagnostic/Control ID"   << ", "
    << std::right << std::setw(60) << "Pre-diagnostic/Control file" << ", "
    << std::right << std::setw(18) << "Pre-diag/Control threshold"  << ", "

    << std::right << std::setw( 9) << "Tumour ID"         << ", "
    << std::right << std::setw(17) << "Tumour image ID"   << ", "
    << std::right << std::setw(17) << "Tumour center (x)" << ", "
    << std::right << std::setw(17) << "Tumour center (y)" << ", "

    << std::right << std::setw(11) << "Patch size" << ", "

    << std::right << std::setw(22) << "Pre-diag/Control patch number" << ", "
    << std::right << std::setw(15) << "Patch index (x)" << ", "
    << std::right << std::setw(15) << "Patch index (y)" << ", "
    << std::right << std::setw(22) << "Pre-diag/Control patch density"

    << std::endl;
};


// -----------------------------------------------------------------------------------
/** \brief Calculates the density of regions on interest across a mammogram */
// -----------------------------------------------------------------------------------

int main(int argc, char** argv)
{
  int result = EXIT_SUCCESS;

  std::string dirOutputFullPath;

  std::ofstream *foutOutputDensityCSV = 0;

  const unsigned int   InputDimension = 2;
  typedef float InputPixelType;

  typedef itk::DOMNode::IdentifierType NodeIdentifierType;

  NodeIdentifierType iNode = 0, iNodeRecord = 0;

  itk::DOMNode::SizeType nChildren;

  itk::DOMNode::Pointer node;
  itk::DOMNode::Pointer nodeRecord;

  itk::DOMTextNode::Pointer textNode;

  std::string strPatientID;
  std::string strImageID, strFilename, strThreshold, strImageOrder, strMammoType;
  std::string strComments, strCase, strSetNumber, strDiagPreDiagOrControl;
  std::string strTumourID, strTumourLeft, strTumourRight, strTumourDiameter;
  std::string strTumourTop, strTumourBottom, strTumourImageID;
  std::string strBreastEdgeID, strXCoord, strYCoord, strBreastEdgeImageID;
  std::string strPectoralID, strPectoralImageID;


  typedef itk::RegionalMammographicDensity<InputPixelType, InputDimension> ROIMammoDensityType;

  std::list< ROIMammoDensityType::Pointer > listOfPatients;
  std::list< ROIMammoDensityType::Pointer >::iterator itPatient;


  // To pass around command line args
  PARSE_ARGS;

  std::cout << std::endl
            << "Input directory: " << dirInput << std::endl
            << "Input image info XML file: " << fileImageXML << std::endl
            << "Input tumour mask info XML file: " << fileTumourXML << std::endl
            << "Input breast edge points XML file: " << fileBreastEdgeXML << std::endl
            << "Input pectoralis XML file: " << filePectoralisLinePointXML << std::endl
            << "Region size in mm: " << regionSizeInMM << std::endl
            << "Affine registration? " << flgRegisterAffine << std::endl
            << "Non-rigid registration? " << flgRegisterNonRigid << std::endl
            << "Ouput directory: " << dirOutput << std::endl
            << "Output CSV file: " << fileOutputDensityCSV << std::endl
            << "Verbose? " << flgVerbose << std::endl
            << "Overwrite? " << flgOverwrite << std::endl
            << "Debug? " << flgDebug << std::endl
            << std::endl;

  // Validate command line args
  if ( ( filePectoralisLinePointXML.length() == 0 ) ||
       ( fileImageXML.length() == 0 ) ||
       ( fileTumourXML.length() == 0 ) ||
       ( fileBreastEdgeXML.length() == 0 ) )
  {
    std::cerr << "ERROR: Input XML file(s) missing" << std::endl;
    return EXIT_FAILURE;
  }


  // Open the output CSV density measurements file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputDensityCSV.length() != 0 ) {
    foutOutputDensityCSV
      = new std::ofstream( fileOutputDensityCSV.c_str(), std::ios::binary );

    if ((! foutOutputDensityCSV) || foutOutputDensityCSV->bad() || foutOutputDensityCSV->fail()) {
      std::cerr << "ERROR: Could not open CSV output file: " << fileOutputDensityCSV << std::endl;
      return EXIT_FAILURE;
    }
  }

  WriteToCSVFile( foutOutputDensityCSV );


  // Read the XML files
  // ~~~~~~~~~~~~~~~~~~

  itk::DOMNode::Pointer domPectoral, domImage, domTumour, domBreastEdge;

  ReadXMLFile( filePectoralisLinePointXML, domPectoral );
  ReadXMLFile( fileImageXML, domImage );
  ReadXMLFile( fileTumourXML, domTumour );
  ReadXMLFile( fileBreastEdgeXML, domBreastEdge );


  // Iterate through the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domImage->GetNumberOfChildren() );
        iNode++ )
  {
    node = domImage->GetChild( iNode );

    std::cout << std::endl << "Image: " << iNode << std::endl;

    if ( flgDebug )
      std::cout << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "id" ) )
      {
        strImageID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "file_name" ) )
      {
        strFilename   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "DensityThreshold" ) )
      {
        strThreshold  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "image_order" ) )
      {
        strImageOrder = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "comments" ) )
      {
        strComments = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "case" ) )
      {
        strDiagPreDiagOrControl = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "setnum" ) )
      {
        strSetNumber = textNode->GetText();
      }
    }

    if (      strDiagPreDiagOrControl == std::string( "1" ) )
    {
      strMammoType = "pre-diagnostic";
    }
    else if ( strDiagPreDiagOrControl == std::string( "2" ) )
    {
      strMammoType = "diagnostic";
    }
    else if ( strDiagPreDiagOrControl == std::string( "3" ) )
    {
      strMammoType = "control";
    }
    else
    {
      strMammoType = "undefined";
      std::cerr << "ERROR: Mammogram type (prediagnostic/diagnostic/control) undefined" << std::endl;
      exit( EXIT_FAILURE );
    }

    strPatientID = fs::path( strFilename ).branch_path().string();

    std::cout << "    patient id:           " << strPatientID            << std::endl
              << "    image id:             " << strImageID              << std::endl
              << "    file_name:            " << strFilename             << std::endl
              << "    threshold:            " << strThreshold            << std::endl
              << "    order:                " << strImageOrder           << std::endl
              << "    comments:             " << strComments             << std::endl
              << "    prediag/diag/control: " << strDiagPreDiagOrControl << std::endl
              << "    setnum:               " << strSetNumber            << std::endl
              << "    type:                 " << strMammoType            << std::endl;

    // Find a patient with this ID

    bool flgFound = false;

    ROIMammoDensityType::Pointer patient = 0;

    for ( itPatient = listOfPatients.begin();
          itPatient != listOfPatients.end();
          itPatient++ )
    {
      if ( (*itPatient)->GetPatientID() == strPatientID )
      {
        patient = (*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound )
    {
      patient = ROIMammoDensityType::New();

      patient->SetPatientID( strPatientID );
      patient->SetInputDirectory( dirInput );
      patient->SetOutputDirectory( dirOutput );
      patient->SetRegionSizeInMM( regionSizeInMM );

      if ( flgOverwrite ) patient->SetOverwriteOn();

      if ( flgRegisterAffine )   patient->SetRegisterOn();
      if ( flgRegisterNonRigid ) patient->SetRegisterNonRigidOn();

      if ( flgVerbose )   patient->SetVerboseOn();
      if ( flgDebug )     patient->SetDebugOn();
    }

    if ( strMammoType == std::string( "diagnostic" ) )
    {
      patient->SetIDDiagnosticImage(   strImageID );
      patient->SetFileDiagnostic(      strFilename );
      patient->SetThresholdDiagnostic( atoi( strThreshold.c_str() ) );
    }
    else if  ( strMammoType == std::string( "pre-diagnostic" ) )
    {
      patient->SetIDPreDiagnosticImage(   strImageID );
      patient->SetFilePreDiagnostic(      strFilename );
      patient->SetThresholdPreDiagnostic( atoi( strThreshold.c_str() ) );
    }
    else if  ( strMammoType == std::string( "control" ) )
    {
      patient->SetIDControlImage(   strImageID );
      patient->SetFileControl(      strFilename );
      patient->SetThresholdControl( atoi( strThreshold.c_str() ) );
    }

    if ( ! flgFound )
    {
      listOfPatients.push_back( patient );
    }
  }


  // Iterate through the tumours
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domTumour->GetNumberOfChildren() );
        iNode++ )
  {
    node = domTumour->GetChild( iNode );

    if ( flgDebug )
      std::cout << std::endl << "Tumour: " << iNode << std::endl
                << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this tumour

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strTumourID = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "left" ) )
      {
        strTumourLeft = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "right" ) )
      {
        strTumourRight  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "top" ) )
      {
        strTumourTop = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "bottom" ) )
      {
        strTumourBottom = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strTumourImageID = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "tumor_diameter" ) )
      {
        strTumourDiameter = textNode->GetText();
      }
    }

    std::cout << "    tumour id: " << std::left << std::setw(6) << strTumourID
              << "    left: "      << std::left << std::setw(6) << strTumourLeft
              << "    right: "     << std::left << std::setw(6) << strTumourRight
              << "    top: "       << std::left << std::setw(6) << strTumourTop
              << "    bottom: "    << std::left << std::setw(6) << strTumourBottom
              << "    image id: "  << std::left << std::setw(6) << strTumourImageID
              << "    diameter: "  << std::left << std::setw(6) << strTumourDiameter
              << std::endl << std::endl;

    // Find a patient with this image ID

    bool flgFound = false;

    ROIMammoDensityType::Pointer patient = 0;

    for ( itPatient = listOfPatients.begin();
          itPatient != listOfPatients.end();
          itPatient++ )
    {
      if ( (*itPatient)->GetIDDiagnosticImage() == strTumourImageID )
      {
        patient = (*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound )
    {
      std::cerr << "ERROR: Failed to find a patient corresponding to image id: "
                << strTumourImageID << std::endl;
      exit( EXIT_FAILURE );
    }

    patient->SetTumourID(      strTumourID );
    patient->SetTumourImageID( strTumourImageID );

    patient->SetTumourLeft(   atoi( strTumourLeft.c_str() ) );
    patient->SetTumourRight(  atoi( strTumourRight.c_str() ) );
    patient->SetTumourTop(    atoi( strTumourTop.c_str() ) );
    patient->SetTumourBottom( atoi( strTumourBottom.c_str() ) );

    patient->SetTumourDiameter( atof( strTumourDiameter.c_str() ) );

    if ( ! flgFound )
    {
      listOfPatients.push_back( patient );
    }
  }


  // Iterate through the breast edge points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << std::endl;

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domBreastEdge->GetNumberOfChildren() );
        iNode++ )
  {
    node = domBreastEdge->GetChild( iNode );

    std::cout << "BreastEdge: " << std::left << std::setw(6) << iNode;

    if ( flgDebug )
      std::cout << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strBreastEdgeID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "x" ) )
      {
        strXCoord   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "y" ) )
      {
        strYCoord  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strBreastEdgeImageID = textNode->GetText();
      }
    }

    std::cout << " breast edge id: "       << std::left << std::setw(6) << strBreastEdgeID
              << " x coord: "              << std::left << std::setw(6) << strXCoord
              << " y coord: "              << std::left << std::setw(6) << strYCoord
              << " breast edge image id: " << std::left << std::setw(6) << strBreastEdgeImageID
              << std::endl;

    // Find a patient with this image ID

    bool flgFound = false;

    ROIMammoDensityType::Pointer patient = 0;

    for ( itPatient = listOfPatients.begin();
          itPatient != listOfPatients.end();
          itPatient++ )
    {
      if ( ( (*itPatient)->GetIDDiagnosticImage() == strBreastEdgeImageID ) ||
           ( (*itPatient)->GetIDPreDiagnosticImage() == strBreastEdgeImageID ) ||
           ( (*itPatient)->GetIDControlImage() == strBreastEdgeImageID ) )
      {
        patient = (*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound )
    {
      std::cerr << "ERROR: Failed to find a patient corresponding to image id: "
                << strBreastEdgeImageID << std::endl;
      exit( EXIT_FAILURE );
    }

    patient->PushBackBreastEdgeCoord( strBreastEdgeImageID,
                                      atoi( strBreastEdgeID.c_str() ),
                                      atoi( strXCoord.c_str() ),
                                      atoi( strYCoord.c_str() ) );
  }


  // Iterate through the pectoral points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0;
        iNode < static_cast<NodeIdentifierType>( domPectoral->GetNumberOfChildren() );
        iNode++ )
  {
    node = domPectoral->GetChild( iNode );

    std::cout << "Pectoral:   " << std::left << std::setw(6) << iNode;

    if ( flgDebug )
      std::cout << "    tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;


    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0;
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() );
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgDebug )
        std::cout << " tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strPectoralID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "x" ) )
      {
        strXCoord   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "y" ) )
      {
        strYCoord  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strPectoralImageID = textNode->GetText();
      }
    }

    std::cout << " pectoral id: "          << std::left << std::setw(6) << strPectoralID
              << " x coord: "              << std::left << std::setw(6) << strXCoord
              << " y coord: "              << std::left << std::setw(6) << strYCoord
              << " breast edge image id: " << std::left << std::setw(6) << strPectoralImageID
              << std::endl;

    // Find a patient with this image ID

    bool flgFound = false;

    ROIMammoDensityType::Pointer patient = 0;

    for ( itPatient = listOfPatients.begin();
          itPatient != listOfPatients.end();
          itPatient++ )
    {
      if ( ( (*itPatient)->GetIDDiagnosticImage() == strPectoralImageID ) ||
           ( (*itPatient)->GetIDPreDiagnosticImage() == strPectoralImageID ) ||
           ( (*itPatient)->GetIDControlImage() == strPectoralImageID ) )
      {
        patient = (*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound )
    {
      std::cerr << "ERROR: Failed to find a patient corresponding to image id: "
                << strPectoralImageID << std::endl;
      exit( EXIT_FAILURE );
    }

    patient->PushBackPectoralCoord( strPectoralImageID,
                                      atoi( strPectoralID.c_str() ),
                                      atoi( strXCoord.c_str() ),
                                      atoi( strYCoord.c_str() ) );
  }


  // Print out the patients
  // ~~~~~~~~~~~~~~~~~~~~~~

  if ( flgDebug )
  {
    for ( itPatient = listOfPatients.begin();
	  itPatient != listOfPatients.end();
	  itPatient++ )
    {
      (*itPatient)->Print();
    }
  }


  // Iterate through the patients analysing each image pair
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  float progress = 0.;
  float iFile = 0.;
  float nFiles = listOfPatients.size();

  for ( itPatient = listOfPatients.begin();
        itPatient != listOfPatients.end();
        itPatient++, iFile += 1. )
  {
    progress = iFile/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    (*itPatient)->LoadImages();

    progress = (iFile + 0.25)/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    try
    {
      if ( ! (*itPatient)->Compute() )
      {
        std::cerr << "WARNING: Could not compute patient: " << iFile << std::endl;

        (*itPatient)->UnloadImages();
        continue;
      }        
    }

    catch (itk::ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not compute patient: " << iFile << std::endl
                << ex << std::endl;

      (*itPatient)->UnloadImages();
      continue;
    }

    progress = (iFile + 0.5)/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    (*itPatient)->Print( flgVerbose );

    if ( foutOutputDensityCSV )
    {
      (*itPatient)->WriteDataToCSVFile( foutOutputDensityCSV );
    }

    progress = (iFile + 0.75)/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;


    (*itPatient)->UnloadImages();
  }

  progress = iFile/nFiles;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;


  // Close the CSV file?
  // ~~~~~~~~~~~~~~~~~~~

  if ( foutOutputDensityCSV )
  {
    foutOutputDensityCSV->close();
    delete foutOutputDensityCSV;
  }

  return result;
}
