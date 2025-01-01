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
    unsigned short t = 4;
    
    init_PIC();
    //init_PWM();
    init_TMR();   
    
    TRISA = 0b00001000;  //RA0 to RA3 set as outputs, RA4 as input
    
    //LATA = 0x00;
    us_delay(1000);

    while(1)
    {
        // PWM runs on P1A --> RA5
        // nothing left to do
        
        PORTAbits.RA5 = 1;
        asm("nop");
        asm("nop");
        asm("nop");
        asm("nop");
      
//        idx=0;
//        while(idx<1)
//            ++idx;
        
        //us_delay(t);

        
        PORTAbits.RA5 = 0;
        asm("nop");
        asm("nop");
        asm("nop");
        asm("nop");     // 38.4 kHz
        asm("nop");     // 33.3 kHz
//        asm("nop");
//        asm("nop");     // 29.3 kHz
//        asm("nop");
//        asm("nop");
//        asm("nop");
//        asm("nop");     // 23.7 kHz
//        asm("nop");
//        asm("nop");
//        asm("nop");
//        asm("nop");
//        asm("nop");
//        asm("nop");     
//        asm("nop");    // 17.8 kHz
//        asm("nop");
//        asm("nop");
//        asm("nop");
//        asm("nop");    // 14.7 kHz
//        asm("nop");     
        

        
        //us_delay(1);      
//        idx=0;
//        while(idx<1)
//            ++idx;
    }   // end of while(1))
    
}   // end of main
