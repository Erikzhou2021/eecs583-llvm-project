#include <stdio.h>

int main() {
    int i = 1 + 1;          // i = 1 + 1
    int j = i + 2;          // j = i + 2
    int k = j + 3;          // k = j + 3
    int l = k + 4;          // l = k + 4
    int m = l + 5;          // m = l + 5
    int n = m + 6;          // n = m + 6
    int o = n + 7;          // o = n + 7
    int p = o + 8;          // p = o + 8
    int q = p + 9;          // q = p + 9
    int r = q + 10;         // r = q + 10
    int s = r + 11;         // s = r + 11
    int t = s + 12;         // t = s + 12
    int u = t + 13;         // u = t + 13
    int v = u + 14;
    int w = v + 15;
    int x = w + 16;
    int y = x + 17;
    int z = y + 18;

    int result = i + j + k + l + m + n + o + p + q + r + s + t + u + v + w + x + y + z;
    printf("%d\n", result);
    return 0;
}