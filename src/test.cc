# include "iostream"

int main (int argc, char ** argv) {
    printf("current file name=  %s\n", "macro.c");
    printf("current function name=  %s\n", __FUNCTION__);
    printf("current line number=      %d\n", 15);
    printf("current date=    %s\n", "Mar 15 2015");
    printf("current time=       %s\n", "12:31:35");
    return 0;
}