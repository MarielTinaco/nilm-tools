#include <math.h>
#include <stdint.h>

void softmax(int8_t* input, float* output, uint32_t len)
{
    float sum = 0.0f;

    // Compute exponentials and sum
    for (uint32_t i = 0; i < len; i++) {
        output[i] = expf((float)input[i]);
        sum += output[i];
    }

    // Normalize the output
    for (uint32_t i = 0; i < len; i++) {
        output[i] /= sum;
    }
}

void argmax_s8_2x5(int8_t* input, int8_t* output)
{
    int i;
    for(i = 0; i < 5; i++) {
        if(input[i] >= input[i + 5])
            output[i] = 0;
        else
            output[i] = 1;
    }
}

void clamp_s8_5x5(int8_t* input, int8_t min, int8_t max)
{
    int i;

    if (min >= max)
        return;

    for(i = 0; i < 25; i++) {
        if(input[i] <= min)
            input[i] = min;
        if(input[i] >= max) 
            input[i] = max;
    }
}