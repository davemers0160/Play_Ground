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

//------------------------------------------------------------------------------
const volatile unsigned char h = 4;
const volatile unsigned char w = 4;

                                    /*              J13,                H3,                F5,                K3 */
volatile uint32_t *matrix[4][4] = { { (uint32_t*)&PORTJ, (uint32_t*)&PORTH, (uint32_t*)&PORTF, (uint32_t*)&PORTK }, \
                                    /*               J5,                J7,                D1,                H7 */
                                    { (uint32_t*)&PORTJ, (uint32_t*)&PORTJ, (uint32_t*)&PORTD, (uint32_t*)&PORTH }, \
                                    /*              H11,                H9,                K4,                K5 */
                                    { (uint32_t*)&PORTH, (uint32_t*)&PORTH, (uint32_t*)&PORTK, (uint32_t*)&PORTK }, \
                                    /*              H15,               D10,                K6,                J3 */
                                    { (uint32_t*)&PORTH, (uint32_t*)&PORTD, (uint32_t*)&PORTK, (uint32_t*)&PORTJ } };

volatile uint32_t off_mask[4][4] = { { 0x02000, 0x00008, 0x00020, 0x00008 }, \
                                     { 0x00020, 0x00080, 0x00002, 0x00080 }, \
                                     { 0x00800, 0x00200, 0x00010, 0x00020 }, \
                                     { 0x08000, 0x00400, 0x00040, 0x00008 } };

volatile uint32_t on_mask[4][4] = { { ~0x02000, ~0x00008, ~0x00020, ~0x00008 }, \
                                    { ~0x00020, ~0x00080, ~0x00002, ~0x00080 }, \
                                    { ~0x00800, ~0x00200, ~0x00010, ~0x00020 }, \
                                    { ~0x08000, ~0x00400, ~0x00040, ~0x00008 } };

//------------------------------------------------------------------------------
void ms_delay(uint32_t delay)
{
    uint32_t idx;
    
    for(idx = 0; idx < delay; ++idx)
    {
        TMR2 = 0;
        while(TMR2 < 9998);
    }
}

// *****************************************************************************
// *****************************************************************************
// Section: Main Entry Point
// *****************************************************************************
// *****************************************************************************

int main ( void )
{
    uint32_t idx, jdx;
    uint32_t count = 8;
    
    uint8_t x[] = {1, 3, 0, 2, 1, 3, 0, 2};
    uint8_t y[] = {0, 0, 1, 1, 2, 2, 3, 3};
    
    
    /* Initialize all modules */
    SYS_Initialize ( NULL );

    PORTHbits.RH0 = 1;

    TMR2_Start();

    
    for(idx = 0; idx<count; ++idx)
    {
        *matrix[y[idx]][x[idx]] = 0xFFFF;
        ms_delay(1);
    }
    
    while ( true )
    {
        /* Maintain state machines of all polled MPLAB Harmony modules. */
        // SYS_Tasks ( );
        
        for(idx = 0; idx<count; ++idx)
        {
            
            PORTHbits.RH0 =~ PORTHbits.RH0;
            
            for(jdx=0; jdx<50; ++jdx)
            {
                *matrix[y[idx]][x[idx]] = on_mask[y[idx]][x[idx]];
                ms_delay(10);
                
                *matrix[y[idx]][x[idx]] = 0xFFFF;
                ms_delay(10);                
            }
        
        }

        //GPIO_PinWrite(GPIO_PIN_RJ3, true);
        //GPIO_RH0_Set();
        //matrix[0] = 0x01;
        //*tmp_h = 1;
//        *matrix[0] = 0x04;
//        *matrix[1] = 0x02;
        //*matrix[2] = 0x04;
        
        
        //PORTJ = 0x08;
        //PORTH = 0x01;
        
        //__delay_ms(100);
//        for(count = 0; count < 20; ++count )
//        {
//            TMR2 = 0;
//            while(TMR2<9999);
//        }
        
        //GPIO_PinWrite(GPIO_PIN_RJ3, false);
        //GPIO_PinClear(GPIO_PIN_RJ3);
        //GPIO_RH0_Clear();
//        *matrix[0] = 0x00;
//        *matrix[1] = 0x00;
        //*matrix[2] = 0x00;
        //tmp_h = 0;
        //PORTJ = 0x00;
        //PORTH = 0x00;
        //*tmp_h = 0;
        //__delay_ms(100);
        
//        for(count = 0; count < 20; ++count )
//        {
//            TMR2 = 0;
//            while(TMR2<9999);
//        }
    }

    /* Execution should not come here during normal operation */

    return ( EXIT_FAILURE );
}


/*******************************************************************************
 End of File
*/

