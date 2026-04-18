
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8
#define EMPTY -1
#define LOCKED -2

__attribute__((reqd_work_group_size(THREADS, 1, 1)))
__kernel void buildTreeKernel(
    __global float* x,
    __global float* y,
    __global float* z,
    __global volatile float* mass,
    __global volatile int* child,
    __global int* start,
    __global volatile float* nodeSize,
    __global volatile int* bottom,
    __global volatile int* maxDepth)
{

    const float radius = nodeSize[NUMBER_OF_NODES];
    const float rootX  = x[NUMBER_OF_NODES];
    const float rootY  = y[NUMBER_OF_NODES];
    const float rootZ  = z[NUMBER_OF_NODES];

    const int stepSize = get_local_size(0) * get_num_groups(0);

    int localMaxD = 1;

    bool newBody = true;
    int bodyIdx = get_global_id(0);
    int node; 
    int childPath;

    while (bodyIdx < NUM_BODIES)
    {
        float bodyX;
        float bodyY;
        float bodyZ;
        float currentR;
        int depth;

        if (newBody)
        {
            newBody = false;
            bodyX = x[bodyIdx];
            bodyY = y[bodyIdx];
            bodyZ = z[bodyIdx];

            node = NUMBER_OF_NODES;
            depth = 1;
            currentR = radius;
            childPath = 0;

            if (rootX < bodyX)
            {
                childPath += 1;
            }
            if (rootY < bodyY)
            {
                childPath += 2;
            }
            if (rootZ < bodyZ)
            {
                childPath += 4;
            }
        
        }

        int c = child[NUMBER_OF_CELLS * node + childPath];

        while (c >= NUM_BODIES)
        {
            node = c;
            ++depth;
            currentR *= 0.5f;
            childPath = 0;
            if (x[node] < bodyX)
            {
                childPath  = 1;
            } 
            if (y[node] < bodyY)
            {
                childPath += 2;
            } 
            if (z[node] < bodyZ)
            {
                childPath += 4;
            }

            c = child[NUMBER_OF_CELLS * node + childPath]; 
        }

        if (c != LOCKED)
        {
            int locked = NUMBER_OF_CELLS * node + childPath;

            if (c == atom_cmpxchg(&child[locked], c, -2))
            {
                if (c == EMPTY)
                {
                    child[locked] = bodyIdx;
                }
                else
                {
                    int patch = -1;
                    bool forceInserted = false;
                    do
                    {
                        depth++;

                        if (currentR < 0.01f || depth > 52)
                        {
                            child[NUMBER_OF_CELLS * node + childPath] = bodyIdx;
                            mem_fence(CLK_GLOBAL_MEM_FENCE);
                            child[locked] = (patch >= 0) ? patch : c;
                            bool forceInserted = true;
                            goto next_body;
                        }
                        
                        const int cell = atom_dec(bottom) - 1;

                        if (cell <= NUM_BODIES)
                        {   
                            *bottom = NUMBER_OF_NODES;
                            child[locked] = c;
                            bool forceInserted = true;  
                            return;
                        }


                        patch = max(patch, cell);

                        float offX = (float)((childPath & 1)) * currentR;
                        float offY = (float)((childPath >> 1) & 1) * currentR;
                        float offZ = (float)((childPath >> 2) & 1) * currentR;

                        currentR *= 0.5f;

                        nodeSize[cell] = currentR; 

                        offX = x[cell] = x[node] - currentR + offX;
                        offY = y[cell] = y[node] - currentR + offY;
                        offZ = z[cell] = z[node] - currentR + offZ;

                        mass[cell] = -1.0f;
                        start[cell] = -1;    

#pragma unroll NUMBER_OF_CELLS
                        for (int k = 0; k < NUMBER_OF_CELLS; k++)
                        {
                            child[NUMBER_OF_CELLS * cell + k] = -1;
                        } 

                        if (patch != cell)
                        {
                            child[NUMBER_OF_CELLS * node + childPath] = cell;
                        }

                        int exOct = 0;

                        if (offX < x[c])
                        {
                            exOct = 1;
                        } 
                        if (offY < y[c])
                        {
                            exOct += 2;
                        } 
                        if (offZ < z[c]) 
                        {
                            exOct += 4;
                        }

                        child[NUMBER_OF_CELLS * cell + exOct] = c;

                        node = cell;

                        childPath = 0;

                        if (x[cell] < bodyX)
                        {
                            childPath = 1;
                        } 
                        if (y[cell] < bodyY)
                        {
                            childPath += 2;
                        } 
                        if (z[cell] < bodyZ)
                        {
                            childPath += 4;
                        }

                        c = child[NUMBER_OF_CELLS * node + childPath];

                    }while(c >= 0);

                    if(!forceInserted)
                    {
                        child[NUMBER_OF_CELLS * node + childPath] = bodyIdx;

                        //atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);
                        mem_fence(CLK_GLOBAL_MEM_FENCE);

                        child[locked] = patch;
                    }
                }
                next_body:
                localMaxD = max(depth, localMaxD);
                bodyIdx += stepSize;
                newBody = true;
            }
        }

        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    atom_max(maxDepth, localMaxD);
    
}