#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdbool.h> 
/*/Standard bool might not be standard in the project at all*/

/*/debug area*/
#include <string.h> 




/* main size macro */
#define BUFF_SIZE 256 



/*int bufferedInput, bufferedOutput; /*just the buffers right here*/ 
char mainBuffer[BUFF_SIZE]; /*main buffer for commands to go to.*/



/*enumeration for a given state */
enum givenStateOfShell 
{
    ENV = 0,
    COMMAND,
    EOF_CONDITION,
    MOOT,
    ERROR


};





int grabInput( int inputFileDescriptor, char *passedStringBuffer, const int sizeOfString)
{
    /*this function grabs */
    ssize_t stackReadSizeBuffer = read(inputFileDescriptor, (void *) passedStringBuffer, sizeOfString);


    if (stackReadSizeBuffer == -1)
    {
        perror("We had an issue grabbing input from the terminal"); /*use perror for error reporting */
        return -1;
    }
    else
    {
        /*make sure to cast the data type before returning */
        return (ssize_t) stackReadSizeBuffer; 


    }

}



int alterState(char *commandString)
{

    register char myDelim = ' '; 
    char *myInternalString =  strtok(commandString, (const char *) myDelim); /* just with a space delimiter to get the tokens */
    
    return -1; /*issue */
}


int main( int argc, char *argv[], char *envp[] )
{

    if (argc <= 0)
    {

        /*perror("Simple shell!")*/ 

        goto LOOP; /* goto statement to try and fix somethings here */
        



    }

    /*main function */
    int stateVariable = 0; /*just default get the environment */
    bool conditionalVariable = true; /*just to have a condition on an infinite loop*/
    char *pathTable[BUFF_SIZE]; /*A char that has all of the paths in an array of array*/  

    grabInput(STDIN_FILENO, mainBuffer, BUFF_SIZE);
    puts(mainBuffer); /*just print the main buffer*/ 

    return 0; /*short circuit*/


    /*infinite loopk*/

    while (conditionalVariable)
    {
        LOOP:
        /*
        grabInput();
        alterState();
        */ 

        /*Try different */
            switch(stateVariable)
            {



            }
    
    }






}
