/*
 * File:   main.c
 *
 * Created on June 2, 2016, 12:29 PM
 */

#include <xc.h>
#include <stdio.h>
#include "PIC12F1822_Config.h"

//------------------------------------------------------------------------------
#pragma config CLKOUTEN = OFF, MCLRE=OFF, WDTE=OFF, BOREN=OFF, PWRTE=OFF, FCMEN=OFF, IESO=OFF, CP=OFF, CPD=OFF, FOSC=INTOSC
#pragma config STVREN = OFF, BORV = HI, LVP = OFF, WRT = OFF, PLLEN = ON


const unsigned char ON = 0;
const unsigned char OFF = 1;


//void __interrupt() ISR(void)        // interrupt function 
//{
//    // check for the receive UART interrupt
//    if(RCIF == 1)
//    {
//        
//    }
// 
//}


//------------------------------------------------------------------------------
void main(void) 
{

    unsigned short idx;
    
    init_PIC();
    init_PWM();
    init_TMR();   
    
    TRISA = 0b00001000;  //RA0 to RA3 set as outputs, RA4 as input
    
    LATA = 0x00;
    us_delay(1000);

    
    while(1)
    {
        // PWM runs on P1A --> RA5
        // nothing left to do
        
        us_delay(100);

    }   // end of while(1))
    
}   // end of main
