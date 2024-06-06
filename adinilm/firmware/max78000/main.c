#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"
#include "sensor_emul.h"
#include "arm_nnfunctions.h"
#include "nn_ops.h"
#include "parameters.h"

#if (EMUL_FIFO_BITLEN == 8)
uint8_t sensor_fifo[NILM_FIFO_SIZE];
#else

// 1-channel 100x1 data input (100 bytes / 25 32-bit words):
// HWC 100x1, channels 0 to 0
uint32_t sensor_fifo[NILM_FIFO_SIZE];

#endif

volatile uint32_t cnn_time; // Stopwatch

static int8_t output_states[10];
static int8_t output_rms[25];

#if (NILM_USE_INFERENCE_BUFFER == 1)
static int8_t states_a0[NILM_FIFO_SIZE]; 	// Discrete states signal of appliance 0
static int8_t states_a1[NILM_FIFO_SIZE]; 	// Discrete states signal of appliance 1
static int8_t states_a2[NILM_FIFO_SIZE]; 	// Discrete states signal of appliance 2
static int8_t states_a3[NILM_FIFO_SIZE]; 	// Discrete states signal of appliance 3
static int8_t states_a4[NILM_FIFO_SIZE]; 	// Discrete states signal of appliance 4
static int8_t rms_a0[NILM_FIFO_SIZE]; 		// Discrete rms signal of appliance 0
static int8_t rms_a1[NILM_FIFO_SIZE]; 		// Discrete rms signal of appliance 1
static int8_t rms_a2[NILM_FIFO_SIZE]; 		// Discrete rms signal of appliance 2
static int8_t rms_a3[NILM_FIFO_SIZE]; 		// Discrete rms signal of appliance 3
static int8_t rms_a4[NILM_FIFO_SIZE]; 		// Discrete rms signal of appliance 4

static void session_update_infer_bufs(int8_t * states, int8_t * rms) 
{
	int i;

	for(i = 1; i < NILM_FIFO_SIZE; i++) {
		states_a0[i - 1] = states_a0[i];
		states_a1[i - 1] = states_a1[i];
		states_a2[i - 1] = states_a2[i];
		states_a3[i - 1] = states_a3[i];
		states_a4[i - 1] = states_a4[i];
		rms_a0[i - 1] = rms_a0[i];
		rms_a1[i - 1] = rms_a1[i];
		rms_a2[i - 1] = rms_a2[i];
		rms_a3[i - 1] = rms_a3[i];
		rms_a4[i - 1] = rms_a4[i];
	}

	states_a0[NILM_FIFO_SIZE - 1] = states[0];
	states_a1[NILM_FIFO_SIZE - 1] = states[1];
	states_a2[NILM_FIFO_SIZE - 1] = states[2];
	states_a3[NILM_FIFO_SIZE - 1] = states[3];
	states_a4[NILM_FIFO_SIZE - 1] = states[4];
	rms_a0[NILM_FIFO_SIZE - 1] = rms[0];
	rms_a1[NILM_FIFO_SIZE - 1] = rms[1];
	rms_a2[NILM_FIFO_SIZE - 1] = rms[2];
	rms_a3[NILM_FIFO_SIZE - 1] = rms[3];
	rms_a4[NILM_FIFO_SIZE - 1] = rms[4];
}

#else

static void session_update_infer_bufs(int8_t * states, int8_t * rms) 
{
	return;
}

#endif

int session_load_input(void) 
{
	memcpy32((uint32_t *) 0x50400000, sensor_fifo, NILM_FIFO_SIZE);
	return 0;
}

int session_init(void)
{
	#if (NILM_USE_INFERENCE_BUFFER == 1)
	memset(states_a0, 0, NILM_FIFO_SIZE);
	memset(states_a1, 0, NILM_FIFO_SIZE);
	memset(states_a2, 0, NILM_FIFO_SIZE);
	memset(states_a3, 0, NILM_FIFO_SIZE);
	memset(states_a4, 0, NILM_FIFO_SIZE);
	memset(rms_a0, 0, NILM_FIFO_SIZE);
	memset(rms_a1, 0, NILM_FIFO_SIZE);
	memset(rms_a2, 0, NILM_FIFO_SIZE);
	memset(rms_a3, 0, NILM_FIFO_SIZE);
	memset(rms_a4, 0, NILM_FIFO_SIZE);
	#endif

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

#if(NILM_OUTPUT_ALL == 1)

int session_fetch_output(void)
{
	int i, j, k;
	int8_t softmax_input_vec[2];
	int8_t softmax_output_vec[2];
	#if (NILM_USE_INFERENCE_BUFFER == 1)
	int8_t states[5] = {0};
	#endif

	cnn_unload(output_states, output_rms);

	k = 0;
	sensor_emul_write("\n\tOUTPUT STATES: \n");
	for (i = 0; i < 2; i++) {
		sensor_emul_write("\t\t[ ");
		for (j = 0; j < 5; j++) 
			sensor_emul_write("%d ", output_states[k++]);
		sensor_emul_write("]\n");
	}

	for (i = 0; i < 5; i++) {
		softmax_input_vec[0] = output_states[i];
		softmax_input_vec[1] = output_states[i + 5];
		arm_softmax_s8(softmax_input_vec, 1, 2, 1024, 8, -64, softmax_output_vec);
		output_states[i] = softmax_output_vec[0];
		output_states[i + 5] = softmax_output_vec[1];
	}

	k = 0;
	sensor_emul_write("\n\tOUTPUT STATES SOFTMAX: \n");
	for (i = 0; i < 2; i++) {
		sensor_emul_write("\t\t[ ");
		for (j = 0; j < 5; j++) 
			sensor_emul_write("%d ", output_states[k++]);
		sensor_emul_write("]\n");
	}

	k = 0;
	sensor_emul_write("\n\tOUTPUT RMS: \n");
	for (i = 0; i < 5; i++) {
		sensor_emul_write("\t\t[ ");
		for (j = 0; j < 5; j++) 
			sensor_emul_write("%d ", output_rms[k++]);
		sensor_emul_write("]\n");
	}

	sensor_emul_write("\t\nTIME: %d us\n", cnn_time);

	// Can add softmax here but does not change output.
	argmax_s8_2x5(output_states, states);
	session_update_infer_bufs(states, &output_rms[10]);

	return 0;
}

#else

int session_fetch_output(void)
{
	int8_t output[10];
	int i;

	cnn_unload(output_states, output_rms);

	// Can add softmax here but does not change output argmax.
	argmax_s8_2x5(output_states, output);
	clamp_s8_5x5(output_rms, 0, 127);
	memcpy(&output[5], &output_rms[10], 5);

	session_update_infer_bufs(&output[0], &output[5]);

	for (i = 0; i < 10; i++) 
		sensor_emul_write("%d, ", output[i]);
	
	return 0;
}

#endif

int session_exit(void)
{
	cnn_disable();
	return 0;
}

int main(void) 
{
	struct sensor_emul_ctx ctx = {
		.fifo_size = NILM_FIFO_SIZE,
		.fifo = sensor_fifo,
		.init = session_init,
		.load_input = session_load_input,
		.infer = session_infer,
		.fetch_output = session_fetch_output,
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