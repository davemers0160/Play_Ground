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
void main(void) {

    unsigned short idx;
    
    unsigned char mode = 2;

    unsigned short mode_0_on = 1000;
    unsigned short mode_0_off = 300;
    
    unsigned short mode_1_on = 1000;
    unsigned short mode_1_off = 800;    

    unsigned short mode_2_on = 1000;
    unsigned short mode_2_off = 10100;  

    unsigned short mode_3_on = 250;
    unsigned short mode_3_off = 250;  
    
    init_PIC();
    init_TMR();   
    //init_UART();
    
    TRISA = 0b00001000;  //RA0 to RA3 set as outputs, RA4 as input
    TRISC = 0b00100000;  //RC0 to RC3 set as input
    
    LATA = 0xFF;
    LATC = 0xFF;
    us_delay(1000);

    
    while(1)
    {
        switch(mode)
        {
        case 0:
            // cycle through each pin
            LATA = 0b00111110;
            us_delay(mode_0_on);
            LATA = 0b00111111;
            us_delay(mode_0_off);

            LATA = 0b00111101;
            us_delay(mode_0_on);
            LATA = 0b00111111;
            us_delay(mode_0_off);

            LATA = 0b00111011;
            us_delay(mode_0_on);
            LATA = 0b00111111;
            us_delay(mode_0_off);

            // MCLR Pin
//            PORTA = 0b00110111;
//            us_delay(mode_0_on);
//            PORTA = 0b00111111;
//            us_delay(mode_0_off);

            LATA = 0b00101111;
            us_delay(mode_0_on);
            LATA = 0b00111111;
            us_delay(mode_0_off);

            LATA = 0b00011111;
            us_delay(mode_0_on);
            LATA = 0b00111111;
            us_delay(mode_0_off);

            LATC = 0b00111110;
            us_delay(mode_0_on);
            LATC = 0b00111111;
            us_delay(mode_0_off);

            LATC = 0b00111101;
            us_delay(mode_0_on);
            LATC = 0b00111111;
            us_delay(mode_0_off);

            LATC = 0b00111011;
            us_delay(mode_0_on);
            LATC = 0b00111111;
            us_delay(mode_0_off);

            LATC = 0b00110111;
            us_delay(mode_0_on);
            LATC = 0b00111111;
            us_delay(mode_0_off);

            LATC = 0b00101111;
            us_delay(mode_0_on);
            LATC = 0b00111111;
            us_delay(mode_0_off);
            
//            PORTC = 0b00011111;
//            us_delay(mode_0_on);
//            PORTC = 0b00111111;
//            us_delay(mode_0_off);
            
            //PORTC = 0b00111111;
            //us_delay(mode_delay+mode_delay+mode_delay);
            break;
            
        case 1:
            // cycle through each pin
            LATA = 0b00111100;
            us_delay(mode_1_on);
            LATA = 0b00111111;
            us_delay(mode_1_off);

            LATA = 0b00110011;
            us_delay(mode_1_on);
            LATA = 0b00111111;
            us_delay(mode_1_off);

            LATA = 0b00001111;
            us_delay(mode_1_on);
            LATA = 0b00111111;
            us_delay(mode_1_off);

            LATC = 0b00111100;
            us_delay(mode_1_on);
            LATC = 0b00111111;
            us_delay(mode_1_off);

            LATC = 0b00110011;
            us_delay(mode_1_on);
            LATC = 0b00111111;
            us_delay(mode_1_off);

            LATC = 0b00001111;
            us_delay(mode_1_on);
            LATC = 0b00111111;
            us_delay(mode_1_off);
            
            //us_delay(mode_delay+mode_delay);
            break;
            
        case 2:
            
            LATA = 0b00000000;
            LATC = 0b00010000;
            us_delay(mode_2_on);
            
            LATA = 0b00111111;            
            LATC = 0b00111111;            
            us_delay(mode_2_off);
            
            break;
            
        case 3:
            LATA = 0b00000000;
            LATC = 0b00010000;
            us_delay(mode_3_on);
            
            LATA = 0b00111111;            
            LATC = 0b00111111;            
            us_delay(mode_3_off);
                            
            break;
            
        }

    }   // end of while(1))
    
}   // end of main
