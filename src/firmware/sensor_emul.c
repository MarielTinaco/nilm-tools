#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "sensor_emul.h"

mxc_uart_req_t sensor_emul_uart;
uint32_t sensor_emul_input[EMUL_INPUT_SIZE];
uint8_t sensor_emul_uart_tx[UART_TX_LEN];
uint8_t sensor_emul_uart_rx[UART_RX_LEN];
volatile uint8_t sensor_emul_irq_flag = 0;

void sensor_emul_irq_handler(void)
{
	MXC_UART_AsyncHandler(UART_DEV);
	sensor_emul_irq_flag = 1;
}

static uint16_t to_be16(uint8_t *buf)
{
	return buf[1] | ((uint16_t)buf[0] << 8);
}

int sensor_emul_init(void) 
{
	int ret;

	sensor_emul_uart.uart = UART_DEV;
	sensor_emul_uart.txData = sensor_emul_uart_tx;
	sensor_emul_uart.rxData = sensor_emul_uart_rx;
	sensor_emul_uart.txLen = 0;
	sensor_emul_uart.rxLen = UART_RX_LEN;
	sensor_emul_uart.callback = NULL;

	NVIC_ClearPendingIRQ(UART_IRQ);
	NVIC_DisableIRQ(UART_IRQ);
	MXC_NVIC_SetVector(UART_IRQ, sensor_emul_irq_handler);
	NVIC_EnableIRQ(UART_IRQ);

	ret = MXC_UART_Init(UART_DEV, UART_BAUDRATE, MXC_UART_IBRO_CLK);
	if (ret)
		return ret;

	ret = MXC_UART_SetRXThreshold(UART_DEV, 2);
	if (ret)
		return ret;

	ret = MXC_UART_ClearTXFIFO(UART_DEV);
	if (ret)
		return ret;

	ret = MXC_UART_ClearRXFIFO(UART_DEV);
	if (ret)
		return ret;

	return MXC_UART_TransactionAsync(&sensor_emul_uart);
}

int sensor_emul_write(const char *data, ...)
{
#if (UART_DEV_NUM != CONSOLE_UART || !UART_USE_CONSOLE)
	va_list pArgs;
	int len, i;

	if(data != NULL) {
		va_start(pArgs, data);
		len = vsnprintf((char *)sensor_emul_uart.txData, UART_TX_LEN, data, pArgs);
		va_end(pArgs);
	} else 
		return E_INVALID;

	for (i = 0; i < len; i++) {
		MXC_UART_WriteCharacterRaw(sensor_emul_uart.uart, sensor_emul_uart.txData[i]);
		MXC_Delay(100);
	}

	return 0;
#else
	char temp[UART_TX_LEN];
	va_list pArgs;
	if(data != NULL) {
		va_start(pArgs, data);
		vsnprintf(temp, UART_TX_LEN, data, pArgs);
		va_end(pArgs);

		printf(temp);
		return 0;
	} else
		return E_INVALID;
#endif
}

int sensor_emul_reset(void)
{
	int ret;

	sensor_emul_irq_flag = 0;

	memset((void *)sensor_emul_uart.txData, 0, sensor_emul_uart.txLen);
	memset((void *)sensor_emul_uart.rxData, 0, sensor_emul_uart.rxLen);

	ret = MXC_UART_ClearTXFIFO(UART_DEV);
	if (ret)
		return ret;

	ret = MXC_UART_ClearRXFIFO(UART_DEV);
	if (ret)
		return ret;
	
	sensor_emul_uart.txCnt = 0;
	sensor_emul_uart.rxCnt = 0;

	return 0;
}

int sensor_emul_parse(uint32_t *buf)
{
	int i;
	uint32_t count = 0;
	size_t len;
	char token[5] = {0};

	len = to_be16(&sensor_emul_uart_rx[2]);

	for (i = 4; i <= len + 6; i++) {
		if (count >= EMUL_INPUT_SIZE)
			return 0;

		if(sensor_emul_uart_rx[i] == ',' || sensor_emul_uart_rx[i] == 0xdd) {
			buf[count] = atoi(token);
			count++;
			memset(token, 0, 5);

			if (sensor_emul_uart_rx[i] == 0xdd && sensor_emul_uart_rx[i] == 0x01)
				return 0;
		} else 
			strncat(token, (const char *)&sensor_emul_uart_rx[i], 1);
	}

	return 0;
}

int sensor_emul_run(void)
{
	int ret = 0;
	uint32_t command;

	if (sensor_emul_irq_flag != 1)
		return 0;

	if (UART_RX_DELAY)
		MXC_Delay(MXC_DELAY_MSEC(UART_RX_DELAY));

	/* Check if TX/RX is busy */
	if (MXC_UART_GetStatus(sensor_emul_uart.uart) & 0x3) 
		sensor_emul_write("NAK");

	command = to_be16(sensor_emul_uart_rx);

	switch (command) {
		case EMUL_CMD_INIT:
			sensor_emul_write("ACK");
			break;
		case EMUL_CMD_FETCH_DATA:
			ret = sensor_emul_parse(sensor_emul_input);
			if (ret) {
				sensor_emul_write("NAK");
				goto reset;
			}
			sensor_emul_write("ACK");

			/* TODO: insert inference here */
			// idx = cnn_run();
			// sensor_emul_write(classes[idx]);

			sensor_emul_write("TEST_PRED. Last values are %d %d %d %d %d",
                              sensor_emul_input[EMUL_INPUT_SIZE - 5],
                              sensor_emul_input[EMUL_INPUT_SIZE - 4],
                              sensor_emul_input[EMUL_INPUT_SIZE - 3],
                              sensor_emul_input[EMUL_INPUT_SIZE - 2],
                              sensor_emul_input[EMUL_INPUT_SIZE - 1]);
			break;
		default:
			break;
	}

	reset:
	sensor_emul_reset();
	return ret;
}