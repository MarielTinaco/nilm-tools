#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "sensor_emul.h"

mxc_uart_req_t sensor_emul_uart;
uint8_t sensor_emul_uart_tx[UART_TX_LEN];
uint8_t sensor_emul_uart_rx[UART_RX_LEN];
volatile uint8_t sensor_emul_irq_flag = 0;

void sensor_emul_irq_handler(void)
{
	MXC_UART_AsyncHandler(UART_DEV);
	sensor_emul_irq_flag = 1;
}

static uint16_t to_be16(uint8_t * buf)
{
	return buf[1] | ((uint16_t)buf[0] << 8);
}

#if (EMUL_FIFO_BITLEN == 8)
static void update_fifo(uint8_t * buf, size_t size, uint8_t value) 
{
	size_t i;

	for (i = 1; i < size; i++) 
		buf[i-1] = buf[i];
	
	buf[size - 1] = value;
}
#else
static void update_fifo(uint32_t * buf, size_t size, uint8_t value) 
{
	size_t i;

	for (i = 1; i < size; i++) 
		buf[i-1] = buf[i];
	
	buf[size - 1] = value;
}
#endif

int sensor_emul_init(struct sensor_emul_ctx * ctx) 
{
	int ret;

	if (ctx->fifo == NULL || ctx->fifo_size <= 0)
		return -E_INVALID;

	memset(ctx->fifo, 0, ctx->fifo_size);

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

#if (EMUL_FIFO_BITLEN == 8)
int sensor_emul_parse(uint8_t *buf, size_t size)
{
	int i;
	size_t count = 0;
	size_t len;
	char token[5] = {0};

	len = to_be16(&sensor_emul_uart_rx[2]);

	for (i = 4; i <= len + 6; i++) {
		if (count >= EMUL_INPUT_SIZE)
			return 0;

		if(sensor_emul_uart_rx[i] == ',' || sensor_emul_uart_rx[i] == 0xdd) {
			#if (EMUL_FIFO_BITLEN == 8)
			update_fifo(buf, size, (uint8_t)atoi(token));
			#else
			update_fifo(buf, size, (uint32_t)atoi(token));
			#endif

			memset(token, 0, 5);
			count++;

			if (sensor_emul_uart_rx[i] == 0xdd && sensor_emul_uart_rx[i] == 0x01)
				return 0;
		} else 
			strncat(token, (const char *)&sensor_emul_uart_rx[i], 1);
	}

	return 0;
}
#else
int sensor_emul_parse(uint32_t *buf, size_t size)
{
	int i;
	size_t count = 0;
	size_t len;
	char token[5] = {0};

	len = to_be16(&sensor_emul_uart_rx[2]);

	for (i = 4; i <= len + 6; i++) {
		if (count >= EMUL_INPUT_SIZE)
			return 0;

		if(sensor_emul_uart_rx[i] == ',' || sensor_emul_uart_rx[i] == 0xdd) {
			#if (EMUL_FIFO_BITLEN == 8)
			update_fifo(buf, size, (uint8_t)atoi(token));
			#else
			update_fifo(buf, size, (uint32_t)atoi(token));
			#endif

			memset(token, 0, 5);
			count++;

			if (sensor_emul_uart_rx[i] == 0xdd && sensor_emul_uart_rx[i] == 0x01)
				return 0;
		} else 
			strncat(token, (const char *)&sensor_emul_uart_rx[i], 1);
	}

	return 0;
}
#endif

int sensor_emul_run(struct sensor_emul_ctx * ctx)
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
			ret = ctx->init();
			if (ret)
				goto reset;

			sensor_emul_write("ACK");
			break;
		case EMUL_CMD_FETCH_DATA:
			ret = sensor_emul_parse(ctx->fifo, ctx->fifo_size);
			if (ret)
				goto reset;

			ret = ctx->load_input();
			if (ret)
				goto reset;
			
			sensor_emul_write("ACK");
			break;
		case EMUL_CMD_INFER:
			ret = ctx->load_input();
			if (ret)
				goto reset;

			ret = ctx->infer();
			if (ret)
				goto reset;

			sensor_emul_write("ACK");
			break;
		case EMUL_CMD_GET_PREDS:
			ret = ctx->fetch_output();
			if (ret)
				goto reset;
			break;
		default:
			break;
	}

	reset:
	if (ret)
		sensor_emul_write("NAK");
	sensor_emul_reset();

	return ret;
}