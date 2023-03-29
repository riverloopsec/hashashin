#include <stdio.h>
#include <stdlib.h>

void wrapper_printf(const char *data) {
	    printf("%s", data);
}

char *read_data() {
	    char *data = malloc(sizeof(char) * 100);
	        scanf("%s", data);
		    return data;
}

int main() {
	    char *data = read_data();
	        wrapper_printf(data);
		    printf("\n");
		        return 0;
}

