#include <stdio.h>
#include "sensor_emul.h"

int main(void) 
{
	sensor_emul_init();
	while (1) {
		sensor_emul_run(); /* Non-blocking */
	}
	
	return 0;
}