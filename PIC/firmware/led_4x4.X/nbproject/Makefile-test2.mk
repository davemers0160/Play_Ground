#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Include project Makefile
ifeq "${IGNORE_LOCAL}" "TRUE"
# do not include local makefile. User is passing all local related variables already
else
include Makefile
# Include makefile containing local settings
ifeq "$(wildcard nbproject/Makefile-local-test2.mk)" "nbproject/Makefile-local-test2.mk"
include nbproject/Makefile-local-test2.mk
endif
endif

# Environment
MKDIR=gnumkdir -p
RM=rm -f 
MV=mv 
CP=cp 

# Macros
CND_CONF=test2
ifeq ($(TYPE_IMAGE), DEBUG_RUN)
IMAGE_TYPE=debug
OUTPUT_SUFFIX=elf
DEBUGGABLE_SUFFIX=elf
FINAL_IMAGE=${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${OUTPUT_SUFFIX}
else
IMAGE_TYPE=production
OUTPUT_SUFFIX=hex
DEBUGGABLE_SUFFIX=elf
FINAL_IMAGE=${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${OUTPUT_SUFFIX}
endif

ifeq ($(COMPARE_BUILD), true)
COMPARISON_BUILD=-mafrlcsj
else
COMPARISON_BUILD=
endif

# Object Directory
OBJECTDIR=build/${CND_CONF}/${IMAGE_TYPE}

# Distribution Directory
DISTDIR=dist/${CND_CONF}/${IMAGE_TYPE}

# Source Files Quoted if spaced
SOURCEFILES_QUOTED_IF_SPACED=../src/config/test2/peripheral/clk/plib_clk.c ../src/config/test2/peripheral/evic/plib_evic.c ../src/config/test2/peripheral/gpio/plib_gpio.c ../src/config/test2/peripheral/uart/plib_uart1.c ../src/config/test2/stdio/xc32_monitor.c ../src/config/test2/interrupts.c ../src/config/test2/exceptions.c ../src/config/test2/initialization.c ../src/main.c ../src/config/test2/peripheral/tmr/plib_tmr2.c

# Object Files Quoted if spaced
OBJECTFILES_QUOTED_IF_SPACED=${OBJECTDIR}/_ext/1013816111/plib_clk.o ${OBJECTDIR}/_ext/1363459140/plib_evic.o ${OBJECTDIR}/_ext/1363405312/plib_gpio.o ${OBJECTDIR}/_ext/1363002369/plib_uart1.o ${OBJECTDIR}/_ext/173167225/xc32_monitor.o ${OBJECTDIR}/_ext/1602598797/interrupts.o ${OBJECTDIR}/_ext/1602598797/exceptions.o ${OBJECTDIR}/_ext/1602598797/initialization.o ${OBJECTDIR}/_ext/1360937237/main.o ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o
POSSIBLE_DEPFILES=${OBJECTDIR}/_ext/1013816111/plib_clk.o.d ${OBJECTDIR}/_ext/1363459140/plib_evic.o.d ${OBJECTDIR}/_ext/1363405312/plib_gpio.o.d ${OBJECTDIR}/_ext/1363002369/plib_uart1.o.d ${OBJECTDIR}/_ext/173167225/xc32_monitor.o.d ${OBJECTDIR}/_ext/1602598797/interrupts.o.d ${OBJECTDIR}/_ext/1602598797/exceptions.o.d ${OBJECTDIR}/_ext/1602598797/initialization.o.d ${OBJECTDIR}/_ext/1360937237/main.o.d ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o.d

# Object Files
OBJECTFILES=${OBJECTDIR}/_ext/1013816111/plib_clk.o ${OBJECTDIR}/_ext/1363459140/plib_evic.o ${OBJECTDIR}/_ext/1363405312/plib_gpio.o ${OBJECTDIR}/_ext/1363002369/plib_uart1.o ${OBJECTDIR}/_ext/173167225/xc32_monitor.o ${OBJECTDIR}/_ext/1602598797/interrupts.o ${OBJECTDIR}/_ext/1602598797/exceptions.o ${OBJECTDIR}/_ext/1602598797/initialization.o ${OBJECTDIR}/_ext/1360937237/main.o ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o

# Source Files
SOURCEFILES=../src/config/test2/peripheral/clk/plib_clk.c ../src/config/test2/peripheral/evic/plib_evic.c ../src/config/test2/peripheral/gpio/plib_gpio.c ../src/config/test2/peripheral/uart/plib_uart1.c ../src/config/test2/stdio/xc32_monitor.c ../src/config/test2/interrupts.c ../src/config/test2/exceptions.c ../src/config/test2/initialization.c ../src/main.c ../src/config/test2/peripheral/tmr/plib_tmr2.c



CFLAGS=
ASFLAGS=
LDLIBSOPTIONS=

############# Tool locations ##########################################
# If you copy a project from one host to another, the path where the  #
# compiler is installed may be different.                             #
# If you open this project with MPLAB X in the new host, this         #
# makefile will be regenerated and the paths will be corrected.       #
#######################################################################
# fixDeps replaces a bunch of sed/cat/printf statements that slow down the build
FIXDEPS=fixDeps

.build-conf:  ${BUILD_SUBPROJECTS}
ifneq ($(INFORMATION_MESSAGE), )
	@echo $(INFORMATION_MESSAGE)
endif
	${MAKE}  -f nbproject/Makefile-test2.mk ${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${OUTPUT_SUFFIX}

MP_PROCESSOR_OPTION=32MZ2048EFH144
MP_LINKER_FILE_OPTION=,--script="..\src\config\test2\p32MZ2048EFH144.ld"
# ------------------------------------------------------------------------------------
# Rules for buildStep: assemble
ifeq ($(TYPE_IMAGE), DEBUG_RUN)
else
endif

# ------------------------------------------------------------------------------------
# Rules for buildStep: assembleWithPreprocess
ifeq ($(TYPE_IMAGE), DEBUG_RUN)
else
endif

# ------------------------------------------------------------------------------------
# Rules for buildStep: compile
ifeq ($(TYPE_IMAGE), DEBUG_RUN)
${OBJECTDIR}/_ext/1013816111/plib_clk.o: ../src/config/test2/peripheral/clk/plib_clk.c  .generated_files/flags/test2/74147ac11a8dd3f25848722b0f09b93c8d2adcca .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1013816111" 
	@${RM} ${OBJECTDIR}/_ext/1013816111/plib_clk.o.d 
	@${RM} ${OBJECTDIR}/_ext/1013816111/plib_clk.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1013816111/plib_clk.o.d" -o ${OBJECTDIR}/_ext/1013816111/plib_clk.o ../src/config/test2/peripheral/clk/plib_clk.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1363459140/plib_evic.o: ../src/config/test2/peripheral/evic/plib_evic.c  .generated_files/flags/test2/bda5c65d2fe151e1fc847f3cb2494ae1368309d3 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1363459140" 
	@${RM} ${OBJECTDIR}/_ext/1363459140/plib_evic.o.d 
	@${RM} ${OBJECTDIR}/_ext/1363459140/plib_evic.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1363459140/plib_evic.o.d" -o ${OBJECTDIR}/_ext/1363459140/plib_evic.o ../src/config/test2/peripheral/evic/plib_evic.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1363405312/plib_gpio.o: ../src/config/test2/peripheral/gpio/plib_gpio.c  .generated_files/flags/test2/ffff0bd95f2e6e173b0670426cab5d9d5d75a71a .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1363405312" 
	@${RM} ${OBJECTDIR}/_ext/1363405312/plib_gpio.o.d 
	@${RM} ${OBJECTDIR}/_ext/1363405312/plib_gpio.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1363405312/plib_gpio.o.d" -o ${OBJECTDIR}/_ext/1363405312/plib_gpio.o ../src/config/test2/peripheral/gpio/plib_gpio.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1363002369/plib_uart1.o: ../src/config/test2/peripheral/uart/plib_uart1.c  .generated_files/flags/test2/2dec815aaf0cc539e9b44f737479eadd90e1b222 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1363002369" 
	@${RM} ${OBJECTDIR}/_ext/1363002369/plib_uart1.o.d 
	@${RM} ${OBJECTDIR}/_ext/1363002369/plib_uart1.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1363002369/plib_uart1.o.d" -o ${OBJECTDIR}/_ext/1363002369/plib_uart1.o ../src/config/test2/peripheral/uart/plib_uart1.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/173167225/xc32_monitor.o: ../src/config/test2/stdio/xc32_monitor.c  .generated_files/flags/test2/80596bee4a2990ed60e8c1bb00f56396299dd9fa .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/173167225" 
	@${RM} ${OBJECTDIR}/_ext/173167225/xc32_monitor.o.d 
	@${RM} ${OBJECTDIR}/_ext/173167225/xc32_monitor.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/173167225/xc32_monitor.o.d" -o ${OBJECTDIR}/_ext/173167225/xc32_monitor.o ../src/config/test2/stdio/xc32_monitor.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1602598797/interrupts.o: ../src/config/test2/interrupts.c  .generated_files/flags/test2/154917352ac92c71ff3cc8ab34a8ee366a10ffaa .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1602598797" 
	@${RM} ${OBJECTDIR}/_ext/1602598797/interrupts.o.d 
	@${RM} ${OBJECTDIR}/_ext/1602598797/interrupts.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1602598797/interrupts.o.d" -o ${OBJECTDIR}/_ext/1602598797/interrupts.o ../src/config/test2/interrupts.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1602598797/exceptions.o: ../src/config/test2/exceptions.c  .generated_files/flags/test2/5051e6786cceacb8ea5ac8359b319a9d5856460e .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1602598797" 
	@${RM} ${OBJECTDIR}/_ext/1602598797/exceptions.o.d 
	@${RM} ${OBJECTDIR}/_ext/1602598797/exceptions.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1602598797/exceptions.o.d" -o ${OBJECTDIR}/_ext/1602598797/exceptions.o ../src/config/test2/exceptions.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1602598797/initialization.o: ../src/config/test2/initialization.c  .generated_files/flags/test2/356ac07a114e6956910cfc0cb0ef79bb44d5600f .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1602598797" 
	@${RM} ${OBJECTDIR}/_ext/1602598797/initialization.o.d 
	@${RM} ${OBJECTDIR}/_ext/1602598797/initialization.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1602598797/initialization.o.d" -o ${OBJECTDIR}/_ext/1602598797/initialization.o ../src/config/test2/initialization.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1360937237/main.o: ../src/main.c  .generated_files/flags/test2/86fa162222e8e1c188f45c4937322a3130f833c7 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1360937237" 
	@${RM} ${OBJECTDIR}/_ext/1360937237/main.o.d 
	@${RM} ${OBJECTDIR}/_ext/1360937237/main.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1360937237/main.o.d" -o ${OBJECTDIR}/_ext/1360937237/main.o ../src/main.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1013799736/plib_tmr2.o: ../src/config/test2/peripheral/tmr/plib_tmr2.c  .generated_files/flags/test2/544ba0b280e702887c71ce2682b45a7cfe0e8caa .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1013799736" 
	@${RM} ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o.d 
	@${RM} ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE) -g -D__DEBUG -D__MPLAB_DEBUGGER_PK3=1  -fframe-base-loclist  -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1013799736/plib_tmr2.o.d" -o ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o ../src/config/test2/peripheral/tmr/plib_tmr2.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
else
${OBJECTDIR}/_ext/1013816111/plib_clk.o: ../src/config/test2/peripheral/clk/plib_clk.c  .generated_files/flags/test2/4bd3273a78e4b021be55a77be5fb45fde76e9925 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1013816111" 
	@${RM} ${OBJECTDIR}/_ext/1013816111/plib_clk.o.d 
	@${RM} ${OBJECTDIR}/_ext/1013816111/plib_clk.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1013816111/plib_clk.o.d" -o ${OBJECTDIR}/_ext/1013816111/plib_clk.o ../src/config/test2/peripheral/clk/plib_clk.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1363459140/plib_evic.o: ../src/config/test2/peripheral/evic/plib_evic.c  .generated_files/flags/test2/b070b066c3d8ac73227431b73f085025d4e91a72 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1363459140" 
	@${RM} ${OBJECTDIR}/_ext/1363459140/plib_evic.o.d 
	@${RM} ${OBJECTDIR}/_ext/1363459140/plib_evic.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1363459140/plib_evic.o.d" -o ${OBJECTDIR}/_ext/1363459140/plib_evic.o ../src/config/test2/peripheral/evic/plib_evic.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1363405312/plib_gpio.o: ../src/config/test2/peripheral/gpio/plib_gpio.c  .generated_files/flags/test2/a71587614769ddc3c2015cdde0d6520c5829b7c6 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1363405312" 
	@${RM} ${OBJECTDIR}/_ext/1363405312/plib_gpio.o.d 
	@${RM} ${OBJECTDIR}/_ext/1363405312/plib_gpio.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1363405312/plib_gpio.o.d" -o ${OBJECTDIR}/_ext/1363405312/plib_gpio.o ../src/config/test2/peripheral/gpio/plib_gpio.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1363002369/plib_uart1.o: ../src/config/test2/peripheral/uart/plib_uart1.c  .generated_files/flags/test2/e8ae4a4383167da8da4e2ba5143612b1f18f28bd .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1363002369" 
	@${RM} ${OBJECTDIR}/_ext/1363002369/plib_uart1.o.d 
	@${RM} ${OBJECTDIR}/_ext/1363002369/plib_uart1.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1363002369/plib_uart1.o.d" -o ${OBJECTDIR}/_ext/1363002369/plib_uart1.o ../src/config/test2/peripheral/uart/plib_uart1.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/173167225/xc32_monitor.o: ../src/config/test2/stdio/xc32_monitor.c  .generated_files/flags/test2/80b808a5c4cd15d44e5167630cfefc333595993b .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/173167225" 
	@${RM} ${OBJECTDIR}/_ext/173167225/xc32_monitor.o.d 
	@${RM} ${OBJECTDIR}/_ext/173167225/xc32_monitor.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/173167225/xc32_monitor.o.d" -o ${OBJECTDIR}/_ext/173167225/xc32_monitor.o ../src/config/test2/stdio/xc32_monitor.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1602598797/interrupts.o: ../src/config/test2/interrupts.c  .generated_files/flags/test2/60bbcad91950ca0c01be9d3beef654007d71c547 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1602598797" 
	@${RM} ${OBJECTDIR}/_ext/1602598797/interrupts.o.d 
	@${RM} ${OBJECTDIR}/_ext/1602598797/interrupts.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1602598797/interrupts.o.d" -o ${OBJECTDIR}/_ext/1602598797/interrupts.o ../src/config/test2/interrupts.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1602598797/exceptions.o: ../src/config/test2/exceptions.c  .generated_files/flags/test2/40dbb7ac9aa9ad9352c0acf37f674d8ee5aa6475 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1602598797" 
	@${RM} ${OBJECTDIR}/_ext/1602598797/exceptions.o.d 
	@${RM} ${OBJECTDIR}/_ext/1602598797/exceptions.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1602598797/exceptions.o.d" -o ${OBJECTDIR}/_ext/1602598797/exceptions.o ../src/config/test2/exceptions.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1602598797/initialization.o: ../src/config/test2/initialization.c  .generated_files/flags/test2/5eca40bc54fbe9e1d759e1714a5c3f995b7f9373 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1602598797" 
	@${RM} ${OBJECTDIR}/_ext/1602598797/initialization.o.d 
	@${RM} ${OBJECTDIR}/_ext/1602598797/initialization.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1602598797/initialization.o.d" -o ${OBJECTDIR}/_ext/1602598797/initialization.o ../src/config/test2/initialization.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1360937237/main.o: ../src/main.c  .generated_files/flags/test2/287254345a6d60e8953ccd119f0cf62a504bcbb1 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1360937237" 
	@${RM} ${OBJECTDIR}/_ext/1360937237/main.o.d 
	@${RM} ${OBJECTDIR}/_ext/1360937237/main.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1360937237/main.o.d" -o ${OBJECTDIR}/_ext/1360937237/main.o ../src/main.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
${OBJECTDIR}/_ext/1013799736/plib_tmr2.o: ../src/config/test2/peripheral/tmr/plib_tmr2.c  .generated_files/flags/test2/54d764b503f1f5c4ef425781f25ba6d8eb72b6e5 .generated_files/flags/test2/da39a3ee5e6b4b0d3255bfef95601890afd80709
	@${MKDIR} "${OBJECTDIR}/_ext/1013799736" 
	@${RM} ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o.d 
	@${RM} ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o 
	${MP_CC}  $(MP_EXTRA_CC_PRE)  -g -x c -c -mprocessor=$(MP_PROCESSOR_OPTION)  -ffunction-sections -fdata-sections -O1 -I"../src" -I"../src/config/test2" -Wall -MP -MMD -MF "${OBJECTDIR}/_ext/1013799736/plib_tmr2.o.d" -o ${OBJECTDIR}/_ext/1013799736/plib_tmr2.o ../src/config/test2/peripheral/tmr/plib_tmr2.c    -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -mdfp="${DFP_DIR}"  
	
endif

# ------------------------------------------------------------------------------------
# Rules for buildStep: compileCPP
ifeq ($(TYPE_IMAGE), DEBUG_RUN)
else
endif

# ------------------------------------------------------------------------------------
# Rules for buildStep: link
ifeq ($(TYPE_IMAGE), DEBUG_RUN)
${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${OUTPUT_SUFFIX}: ${OBJECTFILES}  nbproject/Makefile-${CND_CONF}.mk    ../src/config/test2/p32MZ2048EFH144.ld
	@${MKDIR} ${DISTDIR} 
	${MP_CC} $(MP_EXTRA_LD_PRE) -g -mdebugger -D__MPLAB_DEBUGGER_PK3=1 -mprocessor=$(MP_PROCESSOR_OPTION)  -o ${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${OUTPUT_SUFFIX} ${OBJECTFILES_QUOTED_IF_SPACED}          -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)   -mreserve=data@0x0:0x37F   -Wl,--defsym=__MPLAB_BUILD=1$(MP_EXTRA_LD_POST)$(MP_LINKER_FILE_OPTION),--defsym=__MPLAB_DEBUG=1,--defsym=__DEBUG=1,-D=__DEBUG_D,--defsym=__MPLAB_DEBUGGER_PK3=1,--defsym=_min_heap_size=512,--gc-sections,--no-code-in-dinit,--no-dinit-in-serial-mem,-Map="${DISTDIR}/${PROJECTNAME}.${IMAGE_TYPE}.map",--memorysummary,${DISTDIR}/memoryfile.xml -mdfp="${DFP_DIR}"
	
else
${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${OUTPUT_SUFFIX}: ${OBJECTFILES}  nbproject/Makefile-${CND_CONF}.mk   ../src/config/test2/p32MZ2048EFH144.ld
	@${MKDIR} ${DISTDIR} 
	${MP_CC} $(MP_EXTRA_LD_PRE)  -mprocessor=$(MP_PROCESSOR_OPTION)  -o ${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${DEBUGGABLE_SUFFIX} ${OBJECTFILES_QUOTED_IF_SPACED}          -DXPRJ_test2=$(CND_CONF)  -legacy-libc  $(COMPARISON_BUILD)  -Wl,--defsym=__MPLAB_BUILD=1$(MP_EXTRA_LD_POST)$(MP_LINKER_FILE_OPTION),--defsym=_min_heap_size=512,--gc-sections,--no-code-in-dinit,--no-dinit-in-serial-mem,-Map="${DISTDIR}/${PROJECTNAME}.${IMAGE_TYPE}.map",--memorysummary,${DISTDIR}/memoryfile.xml -mdfp="${DFP_DIR}"
	${MP_CC_DIR}\\xc32-bin2hex ${DISTDIR}/led_4x4.X.${IMAGE_TYPE}.${DEBUGGABLE_SUFFIX} 
endif


# Subprojects
.build-subprojects:


# Subprojects
.clean-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${OBJECTDIR}
	${RM} -r ${DISTDIR}

# Enable dependency checking
.dep.inc: .depcheck-impl

DEPFILES=$(shell mplabwildcard ${POSSIBLE_DEPFILES})
ifneq (${DEPFILES},)
include ${DEPFILES}
endif
