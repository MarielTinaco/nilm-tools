#include <stdint.h>

void softmax(int8_t* input, float* output, uint32_t len);
void argmax_s8_2x5(int8_t* input, int8_t* output);
void clamp_s8_5x5(int8_t* input, int8_t min, int8_t max);