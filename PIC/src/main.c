/*******************************************************************************
  Main Source File

  Company:
    Microchip Technology Inc.

  File Name:
    main.c

  Summary:
    This file contains the "main" function for a project.

  Description:
    This file contains the "main" function for a project.  The
    "main" function calls the "SYS_Initialize" function to initialize the state
    machines of all modules in the system
 *******************************************************************************/

// *****************************************************************************
// *****************************************************************************
// Section: Included Files
// *****************************************************************************
// *****************************************************************************

#include <stddef.h>                     // Defines NULL
#include <stdbool.h>                    // Defines true
#include <stdlib.h>                     // Defines EXIT_FAILURE
#include "definitions.h"                // SYS function prototypes


// *****************************************************************************
// *****************************************************************************
// Section: Main Entry Point
// *****************************************************************************
// *****************************************************************************

int main ( void )
{
    uint16_t idx;
    
    /* Initialize all modules */
    SYS_Initialize ( NULL );

    uint8_t data[] = { 10, 255, 10, 128 };
    size_t  data_size = 4;
    
    uint8_t burst_count = 16;
    
    // delay time (us) / tick time (us)
    uint32_t delay_count = (uint32_t)(10000.0/21.3333333333333);
    
    // start timer 0
    TC0_TimerStart();
    
    
    while ( true )
    {
        /* Maintain state machines of all polled MPLAB Harmony modules. */
        //SYS_Tasks ( );
        
        for(idx=0; idx<burst_count; ++idx)
        {
            // reset counter
            TC0_Timer32bitCounterSet(0);

            // send data
            SERCOM0_SPI_Write(&data, data_size);
            
            // wait x milliseconds to send again
            // clock tick = 21333.33_ ns (from MCC)
            while(TC0_Timer32bitCounterGet() < delay_count);

        }
        
        // add long sleep here
        
        
    }

    /* Execution should not come here during normal operation */

    return ( EXIT_FAILURE );
}


/*******************************************************************************
 End of File
*/

