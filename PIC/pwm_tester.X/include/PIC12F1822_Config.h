
/* 
 * File:   
 * Author: 
 * Comments:
 * Revision history: 
 */

// This is a guard condition so that contents of this file are not included
// more than once.  
#ifndef PIC12F1822_CONFIG_H
#define	PIC12F1822_CONFIG_H

#include <xc.h> // include processor files - each processor file is guarded.  

void init_PIC(void);
void init_TMR(void);
void init_UART(void);
void init_PWM(void);

void ms_delay(unsigned short delay);
void us_delay(unsigned short delay);


#ifdef	__cplusplus
extern "C" {
#endif /* __cplusplus */

    // TODO If C++ is being used, regular C code needs function names to have C 
    // linkage so the functions can be used by the c code. 

#ifdef	__cplusplus
}
#endif /* __cplusplus */

#endif	/* XC_HEADER_TEMPLATE_H */

