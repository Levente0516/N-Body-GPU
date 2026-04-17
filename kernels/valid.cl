__kernel void validateTreeKernel(
    __global volatile int* child,
    __global volatile float* mass,
    __global int* bottom,
    __global int* errorFlag) // Initialize this to 0 on host
{
    int gid = get_global_id(0);
    int totalNodes = NUMBER_OF_NODES;
    
    // Only check the nodes that were actually used
    if (gid >= *bottom && gid <= totalNodes) 
    {
        for (int i = 0; i < 8; i++) 
        {
            int c = child[gid * 8 + i];
            
            // 1. Check for hanging locks (-2)
            if (c == -2) {
                atomic_inc(errorFlag); // A thread got stuck splitting
            }
            
            // 2. Check for out-of-bounds pointers
            if (c > totalNodes) {
                atomic_inc(errorFlag); // Memory corruption
            }
        }
    }
}