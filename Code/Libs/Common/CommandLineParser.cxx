/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-20 13:20:45 +0000 (Sun, 20 Nov 2011) $
 Revision          : $Revision: 7817 $
 Last modified by  : $Author: mjc $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdarg.h>

#include <iostream>
#include <iomanip>
#include <sstream>
using namespace std;

#include <ConversionUtils.h>
#include <CommandLineParser.h>


namespace niftk
{
#ifndef EXIT_ERROR
#define EXIT_ERROR -1
#endif

#define TRUE 1
#define FALSE 0
#define STREQUAL(a,b) (strcmp(a,b) == 0)

static const char *type[] = {
   "",				/* OPT_SWITCH */
   "(int) ",			/* OPT_INT */
   "(float) ",			/* OPT_FLOAT */
   "(string) ",			/* OPT_STRING */
   "(var) ",			/* OPT_VARIANT */
   "",				/* OPT_MULT */
   "(double) ",			/* OPT_DOUBLE */
   "(long) ",			/* OPT_LONG */
   "(intx2) ",			/* OPT_INTx2 */
   "(intx3) ",			/* OPT_INTx3 */
   "(intx4) ",			/* OPT_INTx4 */
   "(floatx2) ",		/* OPT_FLOATx2 */
   "(floatx3) ",		/* OPT_FLOATx3 */
   "(floatx4) ",		/* OPT_FLOATx4 */
   "(doublex2) ",		/* OPT_DOUBLEx2 */
   "(doublex3) ",		/* OPT_DOUBLEx3 */
   "(doublex4) ",		/* OPT_DOUBLEx4 */
};


/* ------------------------------------------------------------------------
   CommandLineParser Constructor
   ------------------------------------------------------------------------ */

CommandLineParser::CommandLineParser(int argc, char *argv[],
				     struct CommandLineArgumentDescription *ArgList,
				     bool flgVerbose)
{
  std::stringstream sstr;

  m_Argc = argc;
  m_Argv = argv;

  m_flgVerbose = flgVerbose;
  m_ProgramName = argv[0];

  if (m_flgVerbose) {

    sstr << "Executing: ";

    for (int i = 0; i < argc; i++ )
      sstr << argv[i] << " ";

    std::cout << std::endl << sstr.str() << std::endl;
  }

  m_OptionList = ParseArguments(argc, argv, ArgList);
}


/* ------------------------------------------------------------------------
   CommandLineParser Destructor
   ------------------------------------------------------------------------ */

CommandLineParser::~CommandLineParser()
{
  FreeArguments(m_OptionList);
}




/*

NAME:

   PrintHelp() \- Print help or usage message for programme

PROTOTYPE:

   void  PrintHelp(clArgList, flag )
   struct CommandLineArgumentDescription *clArgList;
   int flag;

DESCRIPTION:

   Print help message about programme. 'clArgList' is the help structure
   of the programme containing all command line parsing info.  'Flag'
   can be one of:

\&   HLP_BRIEF   Print usage information only.
\&   HLP_VERBOSE Print full help message.

SEE ALSO:

   ParseArguments()    GetArgument()    FreeArguments()

*/


void CommandLineParser::PrintHelp(struct CommandLineArgumentDescription *clArgList, 
				  int flag)
{
   int length;			/* Length of line so far. */
   int optlen;			/* Length of brief option string. */
   int delayed;			/* Count of delayed ']'s. */
   int newsent;			/* TRUE if new sentence starting. */
   int i;			/* loop counter. */
   const char *cptr;		/* loop pointer to char. */
   const char *sptr;		/* loop pointer to string. */
   struct CommandLineArgumentDescription *opt;		/* loop pointer to option. */
   char optstr[81];		/* Buffer for constructing line of output.  */

   fprintf(stdout,"%s\n%s, %s\n", NIFTK_COPYRIGHT, NIFTK_PLATFORM, NIFTK_VERSION_STRING);

   if (flag & HLP_VERBOSE) {
     for (opt=clArgList; opt->type != OPT_DONE; opt++)
     {
       // Deliberately empty.
     }


      /* If OPT_DONE has helpmsg field set, then it contains a short
	 message describing programme.  Print it if it is present. */

      if (opt->helpmsg) {
	 fputc('\n',stdout);
	 cptr = opt->helpmsg;		/* Programme short description msg. */
	 while (*cptr == ' ')		/* Skip white space at start. */
	    cptr++;
	 newsent = length = 0;
	 while (*cptr) {
	    if (*cptr == '\n') {	/* Do new paragraph/newline. */
	       fputc(*cptr++,stdout);
	       newsent = length = 0;
	    }else if (*cptr == '\t' && length == 0) {
	       fprintf(stdout,"      ");	/* Indent a new line.  */
	       cptr++;
	       length = 6;
	    }else{			/* Print a word. */
					/* Get word length */
	       optlen = strcspn(cptr, " \t\n");
	       if (length + optlen + newsent + 1 > 80) {
		  fputc('\n',stdout);
		  newsent = length = 0;
	       }
	       if (newsent) {		/* Extra space at start of sentence */
		  fputc(' ',stdout);
		  newsent = 0;
		  length++;
	       }
	       fputc(' ',stdout);
	       for (i=0; i<optlen; i++)	/* Print word */
		  fputc(*cptr++, stdout);
	       if (*(cptr-1) == '.' || *(cptr-1) == '!' || *(cptr-1) == '?')
		  newsent = 1;		/* Check if end of sentence */
	       length += optlen + 1;
	    }
	    while (*cptr == ' ')	/* Skip any white space. */
	       cptr++;
	 }
	 if (length) fputc('\n', stdout);
      }
   }

   /*** Construct brief usage message ***/

   /* Print programme name and standard help options. */

   sprintf(optstr, "\n Usage: %s", m_ProgramName);
   strcat(optstr," [-U] [-h]");
   length = strlen(optstr);
   fprintf(stdout, "%s", optstr);

   /* Print options in clArgList (formatting to 80 char wide display). */

   delayed=0;
   for (opt=clArgList; opt->type != OPT_DONE; opt++) {
      optlen = MakeOption(optstr, opt, &delayed);
      length += optlen+1;
      if (length > 80) {
	 fprintf(stdout,"\n       ");
	 length = optlen+7;
      }else
	 fputc(' ', stdout);
      fprintf(stdout, "%s", optstr);
   }

   /* Tidy up at end of brief usage message */

   if (length + delayed > 80)
      fprintf(stdout,"\n       ");
   for (i=0; i<delayed; i++)
      fputc(']', stdout);
   fputc('\n', stdout);

   /*** Print full help message if required ***/

   if (flag & HLP_VERBOSE) {

      /* Standard help options */

      fputc('\n', stdout);
      fprintf(stdout,"       -U\tPrint usage message and exit.\n");
      fprintf(stdout,"       -h\tPrint this help message and exit.\n");

      /* Options in clArgList */

      for (opt=clArgList; opt->type != OPT_DONE; opt++) {
	 switch (opt->type & OPT_TYPEMASK) {
	  case OPT_SWITCH:
	  case OPT_INT:
	  case OPT_LONG:
	  case OPT_FLOAT:
	  case OPT_DOUBLE:
	  case OPT_INTx2:
	  case OPT_INTx3:
	  case OPT_INTx4:
	  case OPT_FLOATx2:
	  case OPT_FLOATx3:
	  case OPT_FLOATx4:
	  case OPT_DOUBLEx2:
	  case OPT_DOUBLEx3:
	  case OPT_DOUBLEx4:
	  case OPT_STRING:
	  case OPT_VARIANT:
	    if (opt->type & OPT_LONELY)	/* Option without a "key" */
	       fprintf(stdout,"       <%s>%c%s%s\n",opt->parm,"\t\t "[strlen(opt->parm)>=5],
			type[opt->type & OPT_TYPEMASK],(char *)opt->helpmsg);
	    else			/* Option with a "key" */
	       fprintf(stdout,"       -%s%c%s%s\n",opt->key,"\t\t "[strlen(opt->key)>=6],
			type[opt->type & OPT_TYPEMASK],(char *)opt->helpmsg);
	    break;
	  case OPT_MULT:
	    for (cptr = opt->parm, sptr=opt->helpmsg; *cptr ; cptr++,sptr++)
	       fprintf(stdout,"       -%s%c%c%s%s\n",opt->key,*cptr,
			    "\t\t "[strlen(opt->key)>=5],type[OPT_SWITCH],sptr);
	    break;
	  case OPT_MORETYPE:
	    break;
	  default:
	    errmsg("Badly formed option list. Fix and recompile me.");
	    break;
	 }
      }
      fputc('\n', stdout);
   }
}



int CommandLineParser::MakeOption(char *str, struct CommandLineArgumentDescription *opt, int *delayed)
{
   char buff[80];
   int i;

   *str = 0;
   if (! (opt->type & OPT_REQ))
      strcat(str,"[");
   if (! (opt->type & OPT_LONELY)) {
      strcat(str,"-");
      strcat(str,opt->key);
   }
   switch (opt->type & OPT_TYPEMASK) 
     {
     case OPT_INT:
     case OPT_LONG:
     case OPT_FLOAT:
     case OPT_DOUBLE:
     case OPT_INTx2:
     case OPT_INTx3:
     case OPT_INTx4:
     case OPT_FLOATx2:
     case OPT_FLOATx3:
     case OPT_FLOATx4:
     case OPT_DOUBLEx2:
     case OPT_DOUBLEx3:
     case OPT_DOUBLEx4:
     case OPT_STRING:
     case OPT_VARIANT:
       if (! (opt->type & OPT_LONELY))
	 strcat(str," ");
       sprintf(buff,"<%s>",opt->parm);
       break;
     case OPT_MULT:
       sprintf(buff,"{%s}",opt->parm);
       break;
     case OPT_SWITCH:
       buff[0] = 0;
       break;
     case OPT_MORETYPE:
       strcpy(buff,opt->parm);
       break;
     default:
       if (opt->parm)		/* Best we can do in error condition! */
	 sprintf(buff," <%s>",opt->parm); /* Error is flagged above */
       else
	 buff[0] = 0;
       break;
     }

   strcat(str,buff);
   if (! (opt->type & OPT_REQ)) {
      if (opt->type & OPT_NIN)
	 *delayed += 1;
      else {
	 for (i=0; i<*delayed; i++)
	    strcat(str,"]");
	 *delayed = 0;
	 strcat(str,"]");
      }
   }
   return strlen(str);
}




/*

NAME:

   ParseArguments() \- Parse arguments on command line

PROTOTYPE:

   struct ml * ParseArguments( argc, argv, clArgList )
   int argc;
   char *argv[];
   struct CommandLineArgumentDescription *clArgList;

DESCRIPTION:

   Parse the arguments on the command line according to the command
   line description given in 'clArgList'.  'Argc' and 'argv' are as
   passed to main().  ParseArguments() parses the
   arguments on the command line and checks that they are consistent
   with the 'clArgList' description.  If any errors are detected it
   prints a message to stdout and exits.  If the command line is
   consistent it returns a list of command in the 'ml' structure list.
   This is not for your perusal, but for passing to the routine
   GetArgument() which returns each argument for you.  Once argument
   processing is complete then call FreeArguments() with the 'ml' structure
   returned by ParseArguments() to free all resources allocated.

   ParseArguments() also manages the options ''-h''  (or ''-help''), ''-U''
   (or ''-Usage'') and prints out help,
   usage or version messages as appropriate and exits the programme
   whenever one of these options is present.

   An example of use is:

\|   #include <stdio.h>
\|   #include <common/opt.h>
\|
\|   struct CommandLineArgumentDescription help[] = {
\|
\|      // This programme can take the option '-b'.
\|      {OPT_SWITCH, "b", NULL,
\|          "File is binary" },
\|
\|      // It always takes a filename argument.
\|      {OPT_STRING | OPT_LONELY | OPT_REQ,  NULL, "filename",
\|          "Input file name." }
\|
\|      // Indicate end of array and programme description.
\|      {OPT_DONE, NULL, NULL,
\|         "Short message about programme. "
\|         "Terminate paragraphs with '\n'. "
\|         "To really force a new line use '\n'. "
\|         "To indent a line use '\t' at start of line. "
\|         "Note space at end of sentences in lines above. ",
\|         "Do not put a '\n' at end of last line." },
\|
\|   };
\|
\|   // This list identifies the arguments by number.
\|
\|   enum {
\|      O_BINARY,
\|      O_FILENAME
\|   }
\|
\|   main(int argc, char *argv[])
\|      char *filename;
\|      int is_binary;
\|      struct ml *clArgList;
\|         ...
\|      
\|      clArgList = ParseArguments(argc,argv,&prog,help);
\|
\|      GetArgument(clArgList, O_BINARY, &is_binary);
\|      GetArgument(clArgList, O_FILENAME, &filename);
\|
\|      FreeArguments(clArgList);
\|
\|      // Rest of programme follows....
\|

   Simple, eh!  See the header file common/opt.h for the desription
   of the structures involved.

SEE ALSO:

   PrintHelp()    GetArgument()    FreeArguments()

*/


struct ml *CommandLineParser::ParseArguments(int argc, char *argv[],
					     struct CommandLineArgumentDescription *clArgList)
{
  int arg,found,i;
  int more_ok;
  struct CommandLineArgumentDescription *opt;
  char *key;
  struct ml *mlist,*ml;
  char str[80];

  mlist = allocmlist(clArgList);

  /* For each argument on command line do */

  for (arg=1; arg<argc; arg++) {

    found = FALSE;		/* Not found yet */

    if (*argv[arg] == '-'		/* If a key then */
	&& *(argv[arg] + 1) != '0'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '1'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '2'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '3'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '4'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '5'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '6'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '7'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '8'	/* numerical keys are not allowed */
	&& *(argv[arg] + 1) != '9'	/* numerical keys are not allowed */
	&& ! STREQUAL(argv[arg],"-")
	&& ! STREQUAL(argv[arg],"--")) {
       
      /* Parse each argument on command line which has a key, that
	    is, it starts off with a '-'. */
       
      key = argv[arg];	/* Pointer to the key */
       
      /* If standard key, eg. '-h', parse it specially. Note:
	     ParseStandardArguments() doesn't return if it is a standard key. */
       
      ParseStandardArguments(key, clArgList);
       
      key++;			// Step past initial '-'

      if (*(argv[arg] + 1) == '-')
	key++;			// And the second if present

      /* Step through each option in programme option list and see
	    if we can find the key there. */

      for (opt = clArgList; (! found) && opt->type != OPT_DONE; opt++) {

	if (opt->type & OPT_LONELY)
	  break;		/* Break, since no more options with a key */

	if ((opt->type & OPT_TYPEMASK) == OPT_MULT) {

	  /* Check key against an OPT_MULT type option */

	  for (i=0; (! found) && i<int(strlen(opt->parm)); i++) {
	    strcpy(str,opt->key);
	    str[strlen(str) + 1] = 0;
	    str[strlen(str)] = opt->parm[i];
	    if (STREQUAL(key,str)) {

	      /* Found - is a OPT_MULT type key  */

	      found = TRUE;
	      if ((ml = locateml(mlist,opt))) {
		if (opt->type & OPT_UNIQUE) {
		  errmsg("Multiple definitions of base option"
			 " '-%s' not allowed.",opt->key);
		}
		ml->num++;
		ml->data.mult[i]++;
	      }
	      else {
		ml = createml(mlist,opt);
		ml->data.mult[i] = 1;
	      }
	    }
	  }

	}
	else {

	  /* Check key against normal command line option */
	      
	  if (STREQUAL(key,opt->key)) {

	    /* Found - is normal type key */

	    found = TRUE;
	    if ((ml = locateml(mlist,opt))) {
		  
	      /* Option already in command line. */
		  
	      if ((opt->type & OPT_TYPEMASK) != OPT_SWITCH) {
		errmsg("Option '%s' multiply specified.",argv[arg]);
	      }

	      ml->num++;
		  
	    }
	    else {

	      /* New option on command line - create new ml. */
		  
	      ml = createml(mlist,opt);
		  
	      /* Most take an argument, go to next argument on
                        command line and read it if necessary. */
		  
	      if ((opt->type & OPT_TYPEMASK) != OPT_SWITCH) {
		    
		arg = incarg(argc,arg);
		ReadParameter(argv, arg, opt->type, ml);
		    
	      }
	    }
	  }
	}
      }

      if (! found) {   /* Keys must be matched to programme
                                   command line specs */
	errmsg("Invalid option '%s'.",argv[arg]);
      }
    }	 

    else {	/* Keys must be matched to programme
				   command line specs */


      /* We have parsed all options taking a key.  All that is left
	    is parameters on command line that stand alone and come
	    consequectively, for example, file names. */

      /* Step through programme command line specs until we get to
	    options at end that don't take a key. */

      more_ok = FALSE;
      for (opt=clArgList; (opt->type != OPT_DONE)
	     && (! (opt->type & OPT_LONELY))
	     && (opt->type != OPT_MORE); opt++ ) ;

      /* If current option is '--' should step past it. */
	 
      if (STREQUAL(argv[arg],"--")) {
	arg++;
	if (arg == argc) {
	  errmsg("Detected '--' at end of command line");
	}
      }

      /* If there are no such options in programme specs, but there
	    is more on command line, then error. */

      if (opt->type == OPT_DONE) {
	errmsg("Extra arguments on command line.  Try '%s -h'.",m_ProgramName);
      }

      /* Step through programme specs and match to what's left on
	    command line. */

      for ( ; arg < argc && opt->type != OPT_DONE; arg++,opt++) {

	ml = createml(mlist,opt);

	switch (opt->type & OPT_TYPEMASK) {

	case OPT_INT:
	case OPT_LONG:
	case OPT_FLOAT:
	case OPT_DOUBLE:
	case OPT_INTx2:
	case OPT_INTx3:
	case OPT_INTx4:
	case OPT_FLOATx2:
	case OPT_FLOATx3:
	case OPT_FLOATx4:
	case OPT_DOUBLEx2:
	case OPT_DOUBLEx3:
	case OPT_DOUBLEx4:
	case OPT_STRING:
	case OPT_VARIANT:

	  ReadParameter(argv, arg, opt->type, ml);
	  break;

	case OPT_MORETYPE:

	  if (more_ok == TRUE) {
	    errmsg("Extra OPT_MORE in opt list. Fix and recompile me.");
	    exit(EXIT_ERROR);
	  }
	  ml->data.along = arg;
	  more_ok = TRUE;
	  break;

	case OPT_SWITCH:
	case OPT_MULT:

	  errmsg("Illegal type with OPT_LONELY. Fix and recompile me.");
	  break;

	default:

	  errmsg("Unrecognised option type. Fix and recompile me.");
	  break;

	}
      }

      /* If still more arguments on command line and nothing left
	    in programme specs then error */

      if ((arg < argc) && (! more_ok)) {
	errmsg("Extra arguments on command line. Try '%s -h'.",m_ProgramName);
	exit(EXIT_ERROR);
      }
    }
  }

  /* Parsed command line - must check that got required parameters. */

  for (opt= clArgList; opt->type != OPT_DONE; opt++) {
    if (opt->type & OPT_REQ) {
      if (! locateml(mlist,opt)) {
	errmsg("Required argument '-%s' missing. Try '%s -h'.",
	       opt->key, m_ProgramName);
	exit(EXIT_ERROR);
      }
    }
    if (opt->type == OPT_MORE) {
      /* No MORE type options specified, insert default value. */
      if (! locateml(mlist,opt)) {
	ml = createml(mlist,opt);
	ml->data.along = argc;
      }
    }
  }

  return mlist;
}


void CommandLineParser::ParseStandardArguments(char *key, struct CommandLineArgumentDescription *clArgList)
{
   if (STREQUAL(key,"-h") || STREQUAL(key,"-help")) {
      PrintHelp(clArgList, HLP_VERBOSE);
      exit(0);
   }
   if (STREQUAL(key,"-U") || STREQUAL(key,"-Usage")) {
      PrintHelp(clArgList, HLP_BRIEF);
      exit(0);
   }
}


void CommandLineParser::chknotopt(char *argv[], int arg)
{
   if (*argv[arg] == '-') {
      if (isdigit(*(argv[arg]+1)))   /* Negative numbers can be okay */
	 return;	
      errmsg("Required parameter for option '%s' missing.",argv[arg-1]);
   }
}



struct ml *CommandLineParser::locateml(struct ml *mlist, struct CommandLineArgumentDescription *opt)
{
  struct ml *ml;

  for (ml = mlist->next; ml; ml=ml->next) {
    if (ml->opt == opt)
	break;
  }
  return ml;
}


struct ml *CommandLineParser::allocmlist(struct CommandLineArgumentDescription *clArgList)
{
  struct ml *mlptr;

  if ((mlptr = new struct ml) == NULL) {
    errmsg("Out of memory.");
  }
  mlptr->opt = clArgList;
  mlptr->next = 0;			/* Clear as yet unused fields. */
  mlptr->num = 0;
  mlptr->data.along = 0;
  return mlptr;
}



struct ml *CommandLineParser::createml(struct ml *mlist, struct CommandLineArgumentDescription *opt)
{
  struct ml *mlptr,*mlprev;

  for (mlptr = mlist; mlptr->next; mlptr = mlptr->next) ;
  mlprev = mlptr;
  if ((mlptr = new struct ml) == NULL) {
    errmsg("Out of memory.");
  }
  mlprev->next = mlptr;
  mlptr->next = NULL;
  mlptr->num = 1;
  mlptr->data.along = 0;
  mlptr->opt = opt;
  if ((opt->type & OPT_TYPEMASK) == OPT_MULT) {
    if ((mlptr->data.mult = (int *)calloc(sizeof(int),strlen(opt->parm)))== NULL){
      errmsg("Out of memory.");
    }
  }
  return mlptr;
}



int CommandLineParser::incarg(int argc, int arg)
{
  arg++;
  if (arg >= argc) {
    errmsg("Unexpected termination to command line.");
  }
  return arg;
}
 



/*

NAME:

   FreeArguments() \- Free ml structure previously allocated by ParseArguments()

PROTOTYPE:

   void  FreeArguments( mlist )
   struct ml *mlist;

DESCRIPTION:

   Free up resources/memory allocated by ParseArguments().

SEE ALSO:

   ParseArguments()

*/

void CommandLineParser::FreeArguments(struct ml *mlist)
{
  struct ml *ml,*next;

  for (ml=mlist->next; ml; ml = next) {
    next = ml->next;
    if ((ml->opt->type & OPT_TYPEMASK) == OPT_MULT)
      delete ml->data.mult;
    delete ml;
  }
}




/*

NAME:

   GetArgument() \- Get an argument from command line.

PROTOTYPE:

   int  GetArgument( optnum, dataptr )
   int optnum;
   void *dataptr;

DESCRIPTION:

   Get back a parameter that is specified on the command line and that
   has been parsed by ParseArguments().  'Mlist' is the ml list returned by
   ParseArguments(), 'optnum' is the index into the 'clArgList' structure
   (see ParseArguments()) to the particular argument you want to check, and
   'dataptr' is a pointer to some memory space that contains enough
   space to receive the parameter you are reading from the command
   line.

   The integer returned from GetArgument() is the number of times the
   parameter was specified on the command line.  Normally it is 0
   (wasn''t specified on command line) or 1 (appeared once).

   For most types of arguments (such as integers, float values, etc.)
   a value is written into the memory pointed to by 'dataptr' if and
   only if the argument is specified on the command line.  This means
   you can initialise the space pointed to by 'dataptr' with a default
   value before calling GetArgument(), which will only be changed if the
   argument is specified.

   The exceptions to the above are for OPT_SWITCH arguments which
   always stores 'true' if the argument is present or 'false' if
   the argument is absent into 'dataptr'.  The OPT_MORE argument also
   always stores a result into 'dataptr', namely the value of 'argc'
   passed into main() is stored into 'dataptr' if no more arguments
   are present on the command line.

   The possible option types and the type that 'dataptr' should be
   are:

\|      Parameter           dataptr type
\|   OPT_SWITCH                bool *
\|   OPT_INT                   int *
\|   OPT_LONG                  long *
\|   OPT_FLOAT                 float *
\|   OPT_DOUBLE                double *
\|   OPT_STRING                char **
\|   OPT_VARIANT               char **
\|   OPT_MULT                  int *
\|   OPT_MORE                  int *
\|   OPT_INTx2                 int **
\|   OPT_INTx3                 int **
\|   OPT_INTx4                 int **
\|   OPT_FLOATx2               float **
\|   OPT_FLOATx3               float **
\|   OPT_FLOATx4               float **
\|   OPT_DOUBLEx2              double **
\|   OPT_DOUBLEx3              double **
\|   OPT_DOUBLEx4              double **

   See ParseArguments() man page for an example of use.

SEE ALSO:

   ParseArguments()

*/

int CommandLineParser::GetArgument(int optnum, void *dataptr)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;
  int i;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    if ((opt->type & OPT_TYPEMASK) == OPT_SWITCH) {
      if (dataptr)
	*(bool *)dataptr = FALSE;
      return 0;
    }
    return 0;
  }
  else {
    if (dataptr) {
      switch (opt->type & OPT_TYPEMASK) {
      case OPT_SWITCH:
	*(bool *)dataptr = TRUE;
	break;
      case OPT_INT:
	*(int *)dataptr = mlptr->data.along;
	break;
      case OPT_LONG:
	*(long *)dataptr = mlptr->data.along;
	break;
      case OPT_FLOAT:
	*(float *)dataptr = mlptr->data.afloat;
	break;
      case OPT_DOUBLE:
	*(double *)dataptr = mlptr->data.afloat;
	break;
      case OPT_INTx2:
      case OPT_INTx3:
      case OPT_INTx4:
	*(int **)dataptr = mlptr->data.pint;
	break;
      case OPT_FLOATx2:
      case OPT_FLOATx3:
      case OPT_FLOATx4:
	*(float **)dataptr = mlptr->data.pfloat;
	break;
      case OPT_DOUBLEx2:
      case OPT_DOUBLEx3:
      case OPT_DOUBLEx4:
	*(double **)dataptr = mlptr->data.pdouble;
	break;
      case OPT_STRING:
      case OPT_VARIANT:
	*(char **)dataptr = mlptr->data.astr;
	break;
      case OPT_MULT:
	for (i=0; mlptr->data.mult[i] == 0; i++) ;
	*(int *)dataptr = (opt->type & OPT_CHARBCK) ? opt->parm[i] : i;
	break;
      case OPT_MORETYPE:
	*(int *)dataptr = mlptr->data.along;
	break;
      default:
	errmsg("Corrupt option list. Fix and recompile me.");
      }
      if (m_flgVerbose) {
	dumpmlptr(mlptr);
      }
    }
  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, bool &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    if ((opt->type & OPT_TYPEMASK) == OPT_SWITCH) {
      value = FALSE;
      return 0;
    }
    return 0;
  }
  else {
    switch (opt->type & OPT_TYPEMASK) {
    case OPT_SWITCH:
      value = TRUE;
      break;
      
    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}


int CommandLineParser::GetArgument(int optnum, unsigned int &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {
    switch (opt->type & OPT_TYPEMASK) {

    case OPT_INT:
      value = (unsigned int) mlptr->data.along;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, int &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;
  int i;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {
    switch (opt->type & OPT_TYPEMASK) {

    case OPT_INT:
      value = mlptr->data.along;
      break;

    case OPT_MULT:
      for (i=0; mlptr->data.mult[i] == 0; i++) ;
      value = (opt->type & OPT_CHARBCK) ? opt->parm[i] : i;
      break;

    case OPT_MORETYPE:
      value = mlptr->data.along;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}


int CommandLineParser::GetArgument(int optnum, int *&value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];

  if ((mlptr = locateml(m_OptionList,opt)) == NULL) 
    return 0;

  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_INTx2:
    case OPT_INTx3:
    case OPT_INTx4:
      value = mlptr->data.pint;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) 
      dumpmlptr(mlptr);

  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, long int &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_LONG:
      value = mlptr->data.along;
      break;
      
    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, float &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_FLOAT:
      value = mlptr->data.afloat;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, float *&value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_FLOATx2:
    case OPT_FLOATx3:
    case OPT_FLOATx4:
      value = mlptr->data.pfloat;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, double &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_DOUBLE:
      value = mlptr->data.afloat;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, double *&value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_DOUBLEx2:
    case OPT_DOUBLEx3:
    case OPT_DOUBLEx4:
      value = mlptr->data.pdouble;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, char *&value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];
  if ((mlptr = locateml(m_OptionList,opt)) == NULL) {
    return 0;
  }
  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_STRING:
    case OPT_VARIANT:
      value = mlptr->data.astr;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  return mlptr->num;
}

int CommandLineParser::GetArgument(int optnum, std::string &value)
{
  struct ml *mlptr;
  struct CommandLineArgumentDescription *opt;

  opt = &m_OptionList->opt[optnum];

  if ((mlptr = locateml(m_OptionList,opt)) == NULL) 
    return 0;

  else {

    switch (opt->type & OPT_TYPEMASK) {

    case OPT_STRING:
    case OPT_VARIANT:
      value = mlptr->data.astr;
      break;

    default:
      errmsg("Corrupt option list. Fix and recompile me.");
    }
    if (m_flgVerbose) {
      dumpmlptr(mlptr);
    }
  }
  
  return mlptr->num;
}



void CommandLineParser::ReadParameter(char *argv[], int arg, int type, struct ml *mlptr)
{
   char *rest;

   switch (type & OPT_TYPEMASK) {
    case OPT_INT:
    case OPT_LONG:
      chknotopt(argv,arg);
      mlptr->data.along = strtol(argv[arg], &rest, 10);
      if (*rest) errmsg("Badly formatted integer number %s",argv[arg]);
      break;
    case OPT_FLOAT:
    case OPT_DOUBLE:
      chknotopt(argv,arg);
      mlptr->data.afloat = strtod(argv[arg], &rest);
      if (*rest) errmsg("Badly formatted real number %s",argv[arg]);
      break;
    case OPT_INTx2:
      chknotopt(argv,arg);
      mlptr->data.pint = get_intx2(argv[arg]);
      break;
    case OPT_INTx3:
      chknotopt(argv,arg);
      mlptr->data.pint = get_intx3(argv[arg]);
      break;
    case OPT_INTx4:
      chknotopt(argv,arg);
      mlptr->data.pint = get_intx4(argv[arg]);
      break;
    case OPT_FLOATx2:
      chknotopt(argv,arg);
      mlptr->data.pfloat = get_floatx2(argv[arg]);
      break;
    case OPT_FLOATx3:
      chknotopt(argv,arg);
      mlptr->data.pfloat = get_floatx3(argv[arg]);
      break;
    case OPT_FLOATx4:
      chknotopt(argv,arg);
      mlptr->data.pfloat = get_floatx4(argv[arg]);
      break;
    case OPT_DOUBLEx2:
      chknotopt(argv,arg);
      mlptr->data.pdouble = get_doublex2(argv[arg]);
      break;
    case OPT_DOUBLEx3:
      chknotopt(argv,arg);
      mlptr->data.pdouble = get_doublex3(argv[arg]);
      break;
    case OPT_DOUBLEx4:
      chknotopt(argv,arg);
      mlptr->data.pdouble = get_doublex4(argv[arg]);
      break;
    case OPT_STRING:
    case OPT_VARIANT:
      mlptr->data.astr = argv[arg];
      break;
    case OPT_SWITCH:
      break;
    default:
      errmsg("Unrecognised option type. Fix and recompile me!");
      break;
   }
}

int *CommandLineParser::get_intx2(char *str)
{
   int *intx2;

   intx2 = new int[2];

   intx2[0] = 0;
   intx2[1] = 0;

   if (sscanf(str, "%d%*1[x,]%d", &intx2[0], &intx2[1]) != 2) 
      errmsg("Incorrectly formatted intx2 option %s, should be '<a>,<b>'.",
	     str);

   return intx2;
}

int *CommandLineParser::get_intx3(char *str)
{
   int *intx3;

   intx3 = new int[3];

   intx3[0] = 0;
   intx3[1] = 0;
   intx3[2] = 0;

   if (sscanf(str,"%d%*1[x,]%d%*1[x,]%d",
	      &intx3[0], &intx3[1], &intx3[2]) != 3) 
      errmsg("Incorrectly formatted intx3 option %s, should be '<a>,<b>,<c>'.",
	     str);

   return intx3;
}

int *CommandLineParser::get_intx4(char *str)
{
   int *intx4;

   intx4 = new int[4];

   intx4[0] = 0;
   intx4[1] = 0;
   intx4[2] = 0;
   intx4[3] = 0;

   if (sscanf(str,"%d%*1[x,]%d%*1[x,]%d%*1[x,]%d",
	      &intx4[0], &intx4[1], &intx4[2], &intx4[3]) != 4) 
      errmsg("Incorrectly formatted intx4 option %s, should be "
	     "'<a>,<b>,<c>,<d>'.", str);

   return intx4;
}

float *CommandLineParser::get_floatx2(char *str)
{
   float *floatx2;

   floatx2 = new float[2];

   floatx2[0] = 0;
   floatx2[1] = 0;

   if (sscanf(str, "%f%*1[x,]%f", &floatx2[0], &floatx2[1]) != 2) 
      errmsg("Incorrectly formatted floatx2 option %s, should be '<a>,<b>'.",
	     str);

   return floatx2;
}

float *CommandLineParser::get_floatx3(char *str)
{
   float *floatx3;

   floatx3 = new float[3];

   floatx3[0] = 0;
   floatx3[1] = 0;
   floatx3[2] = 0;

   if (sscanf(str,"%f%*1[x,]%f%*1[x,]%f",
	      &floatx3[0], &floatx3[1], &floatx3[2]) != 3) 
      errmsg("Incorrectly formatted floatx3 option %s, should be "
	     "'<a>,<b>,<c>'.", str);

   return floatx3;
}

float *CommandLineParser::get_floatx4(char *str)
{
   float *floatx4;

   floatx4 = new float[4];

   floatx4[0] = 0;
   floatx4[1] = 0;
   floatx4[2] = 0;
   floatx4[3] = 0;

   if (sscanf(str,"%f%*1[x,]%f%*1[x,]%f%*1[x,]%f",
	      &floatx4[0], &floatx4[1], &floatx4[2], &floatx4[3]) != 4) 
      errmsg("Incorrectly formatted floatx4 option %s, should be "
	     "'<a>,<b>,<c>,<d>'.", str);

   return floatx4;
}

double *CommandLineParser::get_doublex2(char *str)
{
   double *doublex2;

   doublex2 = new double[2];

   doublex2[0] = 0;
   doublex2[1] = 0;

   if (sscanf(str, "%lf%*1[x,]%lf", &doublex2[0], &doublex2[1]) != 2) 
      errmsg("Incorrectly formatted doublex2 option %s, should be '<a>,<b>'.",
	     str);

   return doublex2;
}

double *CommandLineParser::get_doublex3(char *str)
{
   double *doublex3;

   doublex3 = new double[3];

   doublex3[0] = 0;
   doublex3[1] = 0;
   doublex3[2] = 0;

   if (sscanf(str,"%lf%*1[x,]%lf%*1[x,]%lf",
	      &doublex3[0], &doublex3[1], &doublex3[2]) != 3) 
      errmsg("Incorrectly formatted doublex3 option %s, should be "
	     "'<a>,<b>,<c>'.", str);

   return doublex3;
}

double *CommandLineParser::get_doublex4(char *str)
{
   double *doublex4;

   doublex4 = new double[4];

   doublex4[0] = 0;
   doublex4[1] = 0;
   doublex4[2] = 0;
   doublex4[3] = 0;

   if (sscanf(str,"%lf%*1[x,]%lf%*1[x,]%lf%*1[x,]%lf",
	      &doublex4[0], &doublex4[1], &doublex4[2], &doublex4[3]) != 4) 
      errmsg("Incorrectly formatted doublex4 option %s, should be "
	     "'<a>,<b>,<c>,<d>'.", str);

   return doublex4;
}


void CommandLineParser::errmsg(const char *s, ...)
{
   va_list ap;
   char str[256];

   va_start(ap,s);
   vsnprintf(str,256,s,ap);
   std::cerr << "ERROR: Command line argument: " << str << std::endl;
   va_end(ap);
   exit(1);
}

void CommandLineParser::dumpmlist(void)
{
  struct ml *mlist;
  struct ml *mlptr;

  mlist = m_OptionList->next;
  for (mlptr=mlist; mlptr; mlptr=mlptr->next) 
    dumpmlptr(mlptr);

  std::cout << std::endl;
}


void CommandLineParser::DumpInfoMessage(struct ml *mlptr, const char *argDesc, std::string value)
{
  struct CommandLineArgumentDescription *opt = mlptr->opt;

  std::string message("Command line argument");

  message += " ["; 
  message += argDesc; 
  message += "]"; 

  if (opt->type & OPT_LONELY) {

    if (value.length() > 0) {
      message += ": ";
      message += value;
    }
  }
  else {
    message += " '-"; 
    message += opt->key; 

    if (value.length() > 0) {
      message += "' \t= "; 
      message += value;
    }
    else
      message += "'"; 
  }

  std::cout << message << std::endl;
}


void CommandLineParser::dumpmlptr(struct ml *mlptr)
{
  int i;
  struct CommandLineArgumentDescription *opt;

  opt = mlptr->opt;

  switch (opt->type & OPT_TYPEMASK) {
  case OPT_INT:
    DumpInfoMessage(mlptr, "INT", 
		    niftk::ConvertToString(mlptr->data.along));
    break;
  case OPT_LONG:
    DumpInfoMessage(mlptr, "LONG INT", 
		    niftk::ConvertToString(mlptr->data.along));
    break;
  case OPT_FLOAT:
    DumpInfoMessage(mlptr, "FLOAT", 
		    niftk::ConvertToString(mlptr->data.afloat));
    break;
  case OPT_DOUBLE:
    DumpInfoMessage(mlptr, "DOUBLE", 
		    niftk::ConvertToString(mlptr->data.afloat));
    break;
  case OPT_INTx2:
    DumpInfoMessage(mlptr, "INTx2", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pint[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pint[1])
		    + std::string(")"));
    break;
  case OPT_INTx3:
    DumpInfoMessage(mlptr, "INTx3", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pint[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pint[1]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pint[2])
		    + std::string(")"));
    break;
  case OPT_INTx4:
    DumpInfoMessage(mlptr, "INTx4", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pint[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pint[1]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pint[2]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pint[3])
		    + std::string(")"));
    break;
  case OPT_FLOATx2:
    DumpInfoMessage(mlptr, "FLOATx2", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pfloat[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pfloat[1])
		    + std::string(")"));
    break;
  case OPT_FLOATx3:
    DumpInfoMessage(mlptr, "FLOATx3", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pfloat[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pfloat[1]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pfloat[2])
		    + std::string(")"));
    break;
  case OPT_FLOATx4:
    DumpInfoMessage(mlptr, "FLOATx4", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pfloat[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pfloat[1]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pfloat[2]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pfloat[3])
		    + std::string(")"));
    break;
  case OPT_DOUBLEx2:
    DumpInfoMessage(mlptr, "DOUBLEx2", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pdouble[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pdouble[1])
		    + std::string(")"));
    break;
  case OPT_DOUBLEx3:
    DumpInfoMessage(mlptr, "DOUBLEx3", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pdouble[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pdouble[1]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pdouble[2])
		    + std::string(")"));
    break;
  case OPT_DOUBLEx4:
    DumpInfoMessage(mlptr, "DOUBLEx4", 
		    std::string("(") 
		    + niftk::ConvertToString(mlptr->data.pdouble[0]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pdouble[1]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pdouble[2]) 
		    + ", " + niftk::ConvertToString(mlptr->data.pdouble[3])
		    + std::string(")"));
    break;
  case OPT_STRING:
  case OPT_VARIANT:
    DumpInfoMessage(mlptr, "STRING", std::string("\"") + mlptr->data.astr + std::string("\""));
    break;
  case OPT_MULT:
    for (i=0; i<int(strlen(opt->parm)); i++) 
      DumpInfoMessage(mlptr, "MULT", niftk::ConvertToString(i) + std::string(" \"") + niftk::ConvertToString(mlptr->data.mult[i]) + std::string("\""));
    break;
  case OPT_SWITCH:
    DumpInfoMessage(mlptr, "BOOL", "");
    break;
  case OPT_MORETYPE:
    for (i=mlptr->data.along; i<m_Argc; i++)
      DumpInfoMessage(mlptr, " ... ", m_Argv[i]);
    break;
  default:
    printf("  Unrecognised option.\n");
  }
}
  
} // namespace
