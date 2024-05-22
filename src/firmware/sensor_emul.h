#include "mxc.h"
#include "uart.h"
#include "mxc_delay.h"
#include "mxc_device.h"
#include "mxc_errors.h"
#include "max78000.h"

#ifndef __SENSOR_EMUL_H__
#define __SENSOR_EMUL_H__

/* UART params */
#define UART_DEV_NUM                    0
#define UART_DEV						MXC_UART_GET_UART(UART_DEV_NUM)
#define UART_IRQ						MXC_UART_GET_IRQ(UART_DEV_NUM)
#define UART_BAUDRATE					115200
#define UART_TX_LEN						50
#define UART_RX_LEN						250
#define UART_RX_DELAY                   200     /* Delay set for async transaction to complete */
#define UART_USE_CONSOLE                0       /* Disable to use custom serial */

/* Emulator params */
#define EMUL_INPUT_SIZE                 100

/* Emulator commands */
#define EMUL_CMD_FETCH_DATA				0xAA02
#define EMUL_CMD_INIT					0xAA01

extern mxc_uart_req_t sensor_emul_uart;
extern uint32_t sensor_emul_input[EMUL_INPUT_SIZE];
extern uint8_t sensor_emul_uart_tx[UART_TX_LEN];
extern uint8_t sensor_emul_uart_rx[UART_RX_LEN];
extern volatile uint8_t sensor_emul_irq_flag;

int sensor_emul_init(void);
int sensor_emul_write(const char *data, ...);
int sensor_emul_reset(void);
int sensor_emul_parse(uint32_t *buf);
int sensor_emul_run(void);

#endif