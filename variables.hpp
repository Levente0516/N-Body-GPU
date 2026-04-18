#ifndef VARIABLES_H
#define VARIABLES_H

#define NUM_BODIES   32768//131072 //32768 //
#define THREADS      64
#define WARPSIZE     64
#define SPAWN_RANGE  100000.0f
#define EMPTY        -1
#define LOCKED       -2
#define THETA        0.5f
#define G            5.0f
#define SOFTENING    50.0f
#define DT           1.0f
#define CAMERAZOOM   2
#define MAXDEPTH     64

#define MAX_NODE     (NUM_BODIES * 2 + 8192)

#endif