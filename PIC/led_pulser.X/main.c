/*
 * File:   main.c
 *
 * Created on June 2, 2016, 12:29 PM
 */

#include <xc.h>
#include <stdio.h>
#include "PIC16F1823_Config.h"

//------------------------------------------------------------------------------
#pragma config CLKOUTEN = OFF, MCLRE=OFF, WDTE=OFF, BOREN=OFF, PWRTE=OFF, FCMEN=OFF, IESO=OFF, CP=OFF, CPD=OFF, FOSC=INTOSC
#pragma config STVREN = OFF, BORV = HI, LVP = OFF, WRT = OFF, PLLEN = ON


const unsigned char ON = 0;
const unsigned char OFF = 1;


void __interrupt() ISR(void)        // interrupt function 
{
    // check for the receive UART interrupt
    if(RCIF == 1)
    {
        
    }
 
}


//------------------------------------------------------------------------------
void main(void) {

    unsigned short idx;
    
    unsigned char mode = 0;
    
    init_PIC();
    init_TMR();   
    init_UART();
    
    TRISA = 0b00000000;  //RA0 to RA3 set as outputs, RA4 as input
    TRISC = 0b00100000;  //RC0 to RC3 set as input
    
    PORTA = 0xFF;
    PORTC = 0xFF;
    
    while(1)
    {
        if(mode == 0)
        {
            // cycle through each pin
            PORTA = 0b00111110;
            us_delay(1000);

            PORTA = 0b00111101;
            us_delay(1000);

            PORTA = 0b00111011;
            us_delay(1000);

            PORTA = 0b00110111;
            us_delay(1000);

            PORTA = 0b00101111;
            us_delay(1000);

            PORTA = 0b00011111;
            us_delay(1000);
            PORTA = 0b00111111;

            PORTC = 0b00001110;
            us_delay(1000);

            PORTC = 0b00001101;
            us_delay(1000);

            PORTC = 0b00001011;
            us_delay(1000);

            PORTC = 0b00000111;
            us_delay(1000);
            PORTC = 0b00001111;
            
        }
        else if(mode == 1)
        {
            // cycle through each pin
            PORTA = 0b00111100;
            us_delay(1000);

            PORTA = 0b00110011;
            us_delay(1000);

            PORTA = 0b00001111;
            us_delay(1000);
            PORTA = 0b00111111;

            PORTC = 0b00001100;
            us_delay(1000);

            PORTC = 0b00000011;
            us_delay(1000);
            PORTC = 0b00001111;
                      
        }

    }   // end of while(1))
    
}   // end of main
