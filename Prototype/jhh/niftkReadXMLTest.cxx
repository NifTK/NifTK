/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-24 16:54:16 +0100 (Fri, 24 Sep 2010) $
 Revision          : $Revision: 3944 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "expat.h"
#include <iostream>
using namespace std;


struct niftk::CommandLineArgumentDescription clArgList[] = {
  
  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "input", "An input xml file."},

  {OPT_DONE, NULL, NULL, 
   "Program to test ITK XML code."

  }
};


enum {

  O_INPUT_XML
};




#define BUFFSIZE	8192

char Buff[BUFFSIZE];

int Depth;

void
start(void *data, const char *el, const char **attr) {
  int i;

  for (i = 0; i < Depth; i++)
    printf("  ");

  printf("%s", el);

  for (i = 0; attr[i]; i += 2) {
    printf(" %s='%s'", attr[i], attr[i + 1]);
  }

  printf("\n");
  Depth++;
}  /* End of start handler */

void
end(void *data, const char *el) {
  Depth--;
}  /* End of end handler */



int main(int argc, char** argv)
{
  FILE * in;
  std::string fileInputXML;	// Input XML filename

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_INPUT_XML, fileInputXML);

  if (!(in = fopen (fileInputXML.c_str(), "r"))) {
    printf ("Unable to open input file %s\n", fileInputXML.c_str());
    exit (2);
  }
  

  XML_Parser p = XML_ParserCreate(NULL);

  if (! p) {
    fprintf(stderr, "Couldn't allocate memory for parser\n");
    exit(-1);
  }

  XML_SetElementHandler(p, start, end);

  for (;;) {
    int done;
    int len;

    len = fread(Buff, 1, BUFFSIZE, in);

    if (ferror(stdin)) {
      fprintf(stderr, "Read error\n");
      exit(-1);
    }
    done = feof(in);

    if (! XML_Parse(p, Buff, len, done)) {
      fprintf(stderr, "Parse error at line %d:\n%s\n",
	      XML_GetCurrentLineNumber(p),
	      XML_ErrorString(XML_GetErrorCode(p)));
      exit(-1);
    }

    if (done)
      break;
  }



  std::cout << "Done";
}

