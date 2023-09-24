/*
 * File:   PIC16F1823_Config.c
 *
 * Created on June 2, 2016, 12:35 PM
 */


#include <xc.h>
#include "PIC16F1823_Config.h"

//------------------------------------------------------------------------------
void init_PIC(void)
{
    OSCCON = 0b11110000;            // 4x PLL enabled, set internal oscillator to 8(32)MHz, clock determined by FOSC
    OPTION_REG = 0b10000000;		// Turns internal pull up resistors off
    
    INTCON = 0b00000000;            // Global Interrupt Enable, Peripheral Interrupt Enable
    //PIE1bits.RCIE = 1;
    
    CM1CON0 = 0b00000111;           // Disables internal comparators
    //VRCON = 0b00000000;           // Disables internal Vref
    //WPUDA = 0b00000000;           // Weak Pull Up/Down disabled
    IOCIE = 0b00000000;             // Disable interrupt on change

    ANSELC = 0b00000000;
    ANSELA = 0b00000000;
    DACCON0 = 0b00000000;           //DAC disabled, positive reference source, DACOUT pin enabled, input is Vdd
}

//------------------------------------------------------------------------------
void init_UART(void)
{
    APFCONbits.RXDTSEL = 0;
    APFCONbits.TXCKSEL = 0;
    
    TXSTA = 0b00100100;
    RCSTA = 0b00110000;
    BAUDCON = 0b00001000;
    SPBRG = 68;             // 115,200 - 0.64% error

}

//------------------------------------------------------------------------------
void init_TMR(void)
{
    // TMR0
    OPTION_REGbits.TMR0CS = 0;          // Timer0 Clock Source Select bit
    OPTION_REGbits.TMR0SE = 0;          // Timer0 Source Edge Select bit
    
    // TMR1
    T1CON = 0b01000001;                 // 1:1 Prescale Value, LP oscillator is disabled, Internal clock (FOSC/1), Timer1 on
    T1GCON = 0b00000000;                // delay set to ~1us
    TMR1 = 0;
    
    // TMR2
    T2CON = 0b00000000;                 // TMR2 -> off
    TMR2 = 0;
    
}

//------------------------------------------------------------------------------
void ms_delay(unsigned short delay)
{
    unsigned short i;

    for(i=0; i<delay; i++)
    {
        TMR1 = 0;
        while(TMR1 < 31912);    //3988 for 1:8 prescale with 32MHz FOSC internal clock
    }

}

//------------------------------------------------------------------------------
void us_delay(unsigned short delay)
{
    unsigned short i;

    for(i=0; i<delay; i++)
    {
        TMR1 = 0;
        while(TMR1 < 31);        //for 1:1 prescale with 32MHz internal clock, theoretically should be 32
    } 
}
