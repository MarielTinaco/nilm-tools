# This file can be used to set build configuration
# variables.  These variables are defined in a file called 
# "Makefile" that is located next to this one.

# For instructions on how to use this system, see
# https://github.com/Analog-Devices-MSDK/VSCode-Maxim/tree/develop#build-configuration

# **********************************************************

# Add your config here!

DEBUG=1

# Add app source and includes
IPATH += ./app/include
VPATH += ./app/src

# Add CMSIS-NN required operations
IPATH += ./CMSIS-NN/Include ./CMSIS-NN/Include/Internal
VPATH += ./CMSIS-NN/Source/SoftmaxFunctions ./CMSIS-NN/Source/ReshapeFunctions

# Add No-OS dependencies
IPATH += ./no-OS/drivers/meter/ade9430
IPATH += ./no-OS/drivers/platform/maxim/common
IPATH += ./no-OS/drivers/platform/maxim/max78000
IPATH += ./no-OS/include
IPATH += ./no-OS/drivers/display/nhd_c12832a1z

VPATH += ./no-OS/drivers/meter/ade9430
VPATH += ./no-OS/drivers/platform/maxim/common
VPATH += ./no-OS/drivers/platform/maxim/max78000
VPATH += ./no-OS/drivers/api
VPATH += ./no-OS/util
VPATH += ./no-OS/drivers/display/nhd_c12832a1z