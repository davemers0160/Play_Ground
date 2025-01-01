/*
 * File:   PIC12F1822_Config.c
 *
 * Created on June 2, 2016, 12:35 PM
 */


#include <xc.h>
#include "PIC12F1822_Config.h"

//------------------------------------------------------------------------------
void init_PIC(void)
{
    //OSCCON = 0b11110000;            // 4x PLL enabled, set internal oscillator to 8(32)MHz, clock determined by FOSC
    OSCCONbits.SPLLEN = 1;
    OSCCONbits.IRCF = 14;
    OSCCONbits.SCS = 3;
    
    OSCTUNEbits.TUN = 0;
    
    OPTION_REG = 0b10000000;		// Turns internal pull up resistors off
    
    INTCON = 0b00000000;            // Global Interrupt Enable, Peripheral Interrupt Enable
    //PIE1bits.RCIE = 1;
    
    CM1CON0 = 0b00000111;           // Disables internal comparators
    //VRCON = 0b00000000;           // Disables internal Vref
    //WPUDA = 0b00000000;           // Weak Pull Up/Down disabled
    IOCIE = 0b00000000;             // Disable interrupt on change

    ANSELA = 0b00000000;
    //DACCON0 = 0b00000000;           //DAC disabled, positive reference source, DACOUT pin enabled, input is Vdd
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
    OPTION_REGbits.TMR0CS = 0;              // Timer0 Clock Source Select bit
    OPTION_REGbits.TMR0SE = 0;              // Timer0 Source Edge Select bit
    
    // TMR1
    T1CONbits.TMR1CS = 1;                   // Timer1 clock source is system clock (FOSC)
    T1CONbits.T1CKPS = 0;                   // 1:1 Prescale value
    T1CONbits.nT1SYNC = 1;
    T1CONbits.T1OSCEN = 0;

    //T1GCON = 0b00000000;                    // delay set to ~1us
    T1GCONbits.TMR1GE = 0;
    
    T1CONbits.TMR1ON = 1;
    
    TMR1 = 0;
    
    // TMR2
    T2CONbits.T2OUTPS = 0;                  // Timer2 Output Postscaler ->  1:1 Postscaler
    T2CONbits.T2CKPS = 1;                   // Timer2 Clock Prescale -> Prescaler = 4
    T2CONbits.TMR2ON = 0;                   // Timer 2 On
    
    TMR2 = 0;
    
}

//------------------------------------------------------------------------------
void init_PWM(void)
{
    unsigned short pulse_width = 0;
    
    // CCP1 CONTROL REGISTER 
    CCP1CONbits.CCP1M = 0x0C;             // PWM mode: P1A, P1C active-high; P1B, P1D active-high
    CCP1CONbits.P1M = 0;                    // Single output; P1A modulated; P1B, P1C, P1D assigned as port pins
    CCP1CONbits.DC1B = 0;
    
    // CCP1 AUTO-SHUTDOWN CONTROL REGISTER
    CCP1ASbits.CCP1ASE = 0;                 // CCP1 outputs are operating
    CCP1ASbits.CCP1AS = 0;                  // Auto-shutdown is disabled
    //CCP1ASbits.PSS1AC = 0;
    
    // ENHANCED PWM CONTROL REGISTER
    PWM1CONbits.P1RSEN = 1;                 // 
    PWM1CONbits.P1DC = 0;                   //
    
    // PR2
    PR2 = 199;                              // 49 --> 40kHz @ 32MHz, 1:4 Prescale --- 2000000/pwm freq - 1
                                            // 66 --> 30kHz @ 32MHz, 1:4 Prescale
                                            // 99 --> 20kHz @ 32MHz, 1:4 Prescale
                                            // 199 --> 10kHz @ 32MHz, 1:4 Prescale
                                                                        
    pulse_width = (4*(PR2+1)) >> 1;
    CCPR1L = 100;//(pulse_width >> 2) & 0x00FF;   // duty_cycle 50% --> CCPR1L = (0.5 * (4*(PR2+1))) >> 2
    
}

//------------------------------------------------------------------------------
void ms_delay(unsigned short delay)
{
    unsigned short i;

    for(i=0; i<delay; ++i)
    {
        TMR1 = 0;
        while(TMR1 < 31912);    //3988 for 1:8 prescale with 32MHz FOSC internal clock
    }

}

//------------------------------------------------------------------------------
void us_delay(unsigned short delay)
{
//    unsigned short i;
//
//    for(i=0; i<delay; ++i)
//    {
//        TMR1 = 0;
//        while(TMR1 < 8);        //for 1:1 prescale with 32MHz internal clock, theoretically should be 32
//    } 
    
    TMR1 = 0;
    delay = delay*4;
    
    while(TMR1 < delay);
}
