#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"
#include "sensor_emul.h"

#define SENSOR_FIFO_SIZE			100

static int8_t ml_data32[CNN_NUM_OUTPUTS];

// 100-channel 1x1 data input (100 bytes total / 1 bytes per channel):
// HWC 1x1, channels 0 to 3
// HWC 1x1, channels 52 to 55
static uint32_t input_0[] = SAMPLE_INPUT_0;

// HWC 1x1, channels 4 to 7
// HWC 1x1, channels 56 to 59
static uint32_t input_4[] = SAMPLE_INPUT_4;

// HWC 1x1, channels 8 to 11
// HWC 1x1, channels 60 to 63
static uint32_t input_8[] = SAMPLE_INPUT_8;

// HWC 1x1, channels 12 to 15
// HWC 1x1, channels 64 to 67
static uint32_t input_12[] = SAMPLE_INPUT_12;

// HWC 1x1, channels 16 to 19
// HWC 1x1, channels 68 to 71
static uint32_t input_16[] = SAMPLE_INPUT_16;

// HWC 1x1, channels 20 to 23
// HWC 1x1, channels 72 to 75
static uint32_t input_20[] = SAMPLE_INPUT_20;

// HWC 1x1, channels 24 to 27
// HWC 1x1, channels 76 to 79
static uint32_t input_24[] = SAMPLE_INPUT_24;

// HWC 1x1, channels 28 to 31
// HWC 1x1, channels 80 to 83
static uint32_t input_28[] = SAMPLE_INPUT_28;

// HWC 1x1, channels 32 to 35
// HWC 1x1, channels 84 to 87
static uint32_t input_32[] = SAMPLE_INPUT_32;

// HWC 1x1, channels 36 to 39
// HWC 1x1, channels 88 to 91
static uint32_t input_36[] = SAMPLE_INPUT_36;

// HWC 1x1, channels 40 to 43
// HWC 1x1, channels 92 to 95
static uint32_t input_40[] = SAMPLE_INPUT_40;

// HWC 1x1, channels 44 to 47
// HWC 1x1, channels 96 to 99
static uint32_t input_44[] = SAMPLE_INPUT_44;

// HWC 1x1, channels 48 to 51
// HWC 1x1, channels 100 to 103
static uint32_t input_48[] = SAMPLE_INPUT_48;

uint8_t sensor_fifo[SENSOR_FIFO_SIZE];

volatile uint32_t cnn_time; // Stopwatch

static uint32_t to_be32(uint8_t * buf, size_t size)
{
	int i;
	uint32_t ret = 0;
	for (i = 0; i < size; i++)
		ret |= buf[i] << (8 * (3 - i));

	return ret;
}

int session_load_input(void) 
{
	int i;

	for (i = 0; i < 2; i++) {
		input_0[i] = to_be32(&sensor_fifo[52*i], 4);
		input_4[i] = to_be32(&sensor_fifo[52*i + 4], 4);
		input_8[i] = to_be32(&sensor_fifo[52*i + 8], 4);
		input_12[i] = to_be32(&sensor_fifo[52*i + 12], 4);
		input_16[i] = to_be32(&sensor_fifo[52*i + 16], 4);
		input_20[i] = to_be32(&sensor_fifo[52*i + 20], 4);
		input_24[i] = to_be32(&sensor_fifo[52*i + 24], 4);
		input_28[i] = to_be32(&sensor_fifo[52*i + 28], 4);
		input_32[i] = to_be32(&sensor_fifo[52*i + 32], 4);
		input_36[i] = to_be32(&sensor_fifo[52*i + 36], 4);
		input_40[i] = to_be32(&sensor_fifo[52*i + 40], 4);
		input_44[i] = to_be32(&sensor_fifo[52*i + 44], 4);

		if (i == 0)
			input_48[i] = to_be32(&sensor_fifo[52*i + 48], 4);
		else
			input_48[i] = 0;
	}

	memcpy32((uint32_t *) 0x50400000, input_0, 2);
	memcpy32((uint32_t *) 0x50408000, input_4, 2);
	memcpy32((uint32_t *) 0x50410000, input_8, 2);
	memcpy32((uint32_t *) 0x50418000, input_12, 2);
	memcpy32((uint32_t *) 0x50800000, input_16, 2);
	memcpy32((uint32_t *) 0x50808000, input_20, 2);
	memcpy32((uint32_t *) 0x50810000, input_24, 2);
	memcpy32((uint32_t *) 0x50818000, input_28, 2);
	memcpy32((uint32_t *) 0x50c00000, input_32, 2);
	memcpy32((uint32_t *) 0x50c08000, input_36, 2);
	memcpy32((uint32_t *) 0x50c10000, input_40, 2);
	memcpy32((uint32_t *) 0x50c18000, input_44, 2);
	memcpy32((uint32_t *) 0x51000000, input_48, 2);

	return 0;
}

int session_init(void)
{
	cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
	cnn_init(); // Bring state machine into consistent state
	cnn_load_weights(); // Load kernels
	cnn_load_bias();
	cnn_configure(); // Configure state machine

	return 0;
}

int session_infer(void)
{
	cnn_start(); // Start CNN processing

	while (cnn_time == 0)
		MXC_LP_EnterSleepMode(); // Wait for CNN

	return 0;
}

int session_fetch_output(void)
{
	int i, j, k = 0;
	cnn_unload(ml_data32);

	sensor_emul_write("\n\tOUTPUT TENSOR: \n");
	for (i = 0; i < 7; i++) {
		sensor_emul_write("\t\t[ ");
		for (j = 0; j < 4; j++) 
			sensor_emul_write("%d ", ml_data32[k++]);
		sensor_emul_write("]\n");
	}

	sensor_emul_write("\tTIME: %d us\n", cnn_time);

	return 0;
}

void print_buf(int8_t *buf, size_t imax, size_t jmax) {
	int i, j, k = 0;
	sensor_emul_write("\n");
	for (i = 0; i < imax; i++) {
		sensor_emul_write("[ ");
		for (j = 0; j < jmax; j++) 
			sensor_emul_write("%d ", buf[k++]);
		sensor_emul_write("]\n");
	}
	sensor_emul_write("\n");
}

int session_test_output(void)
{
	int buf_rows = 10;
	
	int8_t output_data[buf_rows*4];
	memset(output_data, 0, buf_rows*4);

	cnn_peek(0x50401000, 0x1000, buf_rows, output_data);
	print_buf(output_data, buf_rows, 4);

	cnn_peek(0x50801000, 0x1000, buf_rows, output_data);
	print_buf(output_data, buf_rows, 4);

	return 0;
}

int session_exit(void)
{
	cnn_disable();
	return 0;
}


int main(void) 
{
	struct sensor_emul_ctx ctx = {
		.fifo_size = SENSOR_FIFO_SIZE,
		.fifo = sensor_fifo,
		.init = session_init,
		.load_input = session_load_input,
		.infer = session_infer,
		.fetch_output = session_test_output,
		.exit = session_exit,
	};

	MXC_ICC_Enable(MXC_ICC0); // Enable cache

	// Switch to 100 MHz clock
	MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
	SystemCoreClockUpdate();

	MXC_Delay(SEC(2)); // Let debugger interrupt if needed

	sensor_emul_init(&ctx);
	while (1) {
		sensor_emul_run(&ctx); /* Non-blocking */
	}
	
	return 0;
}