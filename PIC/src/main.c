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
#include <string.h>
#include <math.h>

#include "definitions.h"                // SYS function prototypes


// *****************************************************************************
// *****************************************************************************
// Section: Main Entry Point
// *****************************************************************************
// *****************************************************************************
//-----------------------------------------------------------------------------
void maximal_length_sequence(const uint16_t N, uint16_t *taps, uint16_t num_taps, uint8_t *seq[], uint32_t *seq_length)
{
    uint32_t idx, jdx;
    uint16_t tmp;
    uint8_t r[25] = {0};

    // initialize the shift register
    //r = (uint8_t*)calloc(N, sizeof(uint8_t));
    r[0] = 1;

    // initialize the sequence to all 0's
    *seq_length = (1 << N) - 1;
    *seq = (uint8_t*)calloc(*seq_length, sizeof(int8_t));

    for (idx = 0; idx < *seq_length; ++idx)
    {
        // sr.insert(sr.end(), rep, amplitude * (2 * r[N - 1] - 1));
        //sr.insert(sr.end(), 1, r[N - 1]);
        (*seq)[idx] = r[N - 1]; 

        tmp = 0;
        for (jdx = 0; jdx < num_taps; ++jdx)
        {
            tmp += r[taps[jdx]];
        }
        tmp = tmp % 2;

        // go through the register and shift everything
        for(jdx=1; jdx<N; ++jdx)
        {
            r[N-jdx] = r[N-jdx-1];
        }
        r[0] = tmp;
        
    }

}   // end of maximal_length_sequence

//-----------------------------------------------------------------------------
void bit2byte_array(uint8_t binary_array[], uint32_t binary_array_len, uint8_t *byte_array[], uint32_t *byte_array_len)
{
    uint32_t idx;
    int32_t byte_index;
    int32_t bit_in_byte_index;
    
    // Calculate the required length of the byte array
    // Ceiling division: (binary_array_len + 7) / 8
    *byte_array_len = (binary_array_len + 7) / 8;

    // Allocate memory for the byte array
    *byte_array = (uint8_t *)calloc(*byte_array_len, sizeof(uint8_t));
//    if (*byte_array == NULL) 
//    {
//        fprintf(stderr, "Memory allocation failed\n");
//        exit(EXIT_FAILURE);
//    }

    // Convert bits to bytes
    for (idx = 0; idx < binary_array_len; ++idx) 
    {
        byte_index = idx / 8;
        bit_in_byte_index = 7 - (idx % 8); // Bits are typically read from MSB to LSB

        if (binary_array[idx] == 1) 
        {
            (*byte_array)[byte_index] |= (1 << bit_in_byte_index);
        }
    }
    
}   // end of bit2byte_array


//-----------------------------------------------------------------------------
int main ( void )
{
    uint32_t idx;
    
    /* Initialize all modules */
    SYS_Initialize ( NULL );

    uint8_t data[] = { 10, 255, 10, 128 };
    size_t  data_size = 4;
    
    uint16_t burst_count = 100;
    
    // delay time (us) / tick time (us)
    uint32_t delay_count = (uint32_t)(10000.0/21.3333333333333);
    
    // sequence parameters
    uint16_t N = 6;
    uint16_t taps[] = {5,4};
    uint16_t num_taps = 2;
    volatile uint8_t *seq;
    uint32_t seq_length = 0;
    volatile uint8_t *byte_array;
    uint32_t byte_array_len = 0;
    
    // generate the sequence
    maximal_length_sequence(N, taps, num_taps, &seq, &seq_length);

    // convert the sequence to an array of bytes
    bit2byte_array(seq, seq_length, &byte_array, &byte_array_len);
  
    // start timer 0
    TC0_TimerStart();
     
    while ( true )
    {
        /* Maintain state machines of all polled MPLAB Harmony modules. */
        //SYS_Tasks ( );
        
        // turn on LED during burst
        PORT_PinWrite(PORT_PIN_PA23, 1);
        
        for(idx=0; idx<burst_count; ++idx)
        {
            // reset counter
            TC0_Timer32bitCounterSet(0);

            // send data
            SERCOM0_SPI_Write(byte_array, byte_array_len);
            
            // wait x milliseconds to send again
            // clock tick = 21333.33_ ns (from MCC)
            while(TC0_Timer32bitCounterGet() < delay_count);

        }
        
        // turn LED off after burst
        PORT_PinWrite(PORT_PIN_PA23, 0);

        // reset counter
        TC0_Timer32bitCounterSet(0);
            
        // add long sleep here - wait for an interrupt to wake from sleep
        PM_StandbyModeEnter();

    }

    /* Execution should not come here during normal operation */

    return ( EXIT_FAILURE );
}


/*******************************************************************************
 End of File
*/

