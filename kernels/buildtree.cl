
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable

#define NUMBER_OF_CELLS 8

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
    __global volatile int* maxDepth,
    __global int* numNodes,
    __global int* parent)
{

    const int   root   = NUMBER_OF_NODES;
    const float radius = nodeSize[root];
    const float rootX  = x[root];
    const float rootY  = y[root];
    const float rootZ  = z[root];

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

            node = root;
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
            depth++;
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
            const int locked = NUMBER_OF_CELLS * node + childPath;

            if (c == atom_cmpxchg(&child[locked], c, -2))
            {
                if (c == EMPTY)
                {
                    child[locked] = bodyIdx;
                    parent[bodyIdx] = node;
                }
                else
                {
                    int patch = -1;
                    do
                    {
                        depth++;
                        
                        const int cell = atom_dec(bottom) - 1;

                        if (cell <= NUM_BODIES)
                        {   
                            *bottom = NUMBER_OF_NODES;
                            //child[locked] = c;
                            return;
                        }
                        parent[cell] = node;
                        patch = max(patch, cell);

                        float offX = (float)((childPath & 1)) * currentR;
                        float offY = (float)((childPath & 2) >> 1) * currentR;
                        float offZ = (float)((childPath & 4) >> 2) * currentR;

                        currentR *= 0.5f;

                        offX = x[cell] = x[node] - currentR + offX;
                        offY = y[cell] = y[node] - currentR + offY;
                        offZ = z[cell] = z[node] - currentR + offZ;
                        mass[cell] = -1.0f;
                        start[cell] = -1;    
                        nodeSize[cell] = currentR;

#pragma unroll NUMBER_OF_CELLS
                        for (int k = 0; k < NUMBER_OF_CELLS; k++)
                        {
                            child[NUMBER_OF_CELLS * cell + k] = EMPTY;
                        } 

                        if (patch != cell)
                        {
                            child[NUMBER_OF_CELLS * node + childPath] = cell;
                        }

                        int exOct = 0;

                        if (offX < x[c])
                        {
                            exOct  = 1;
                        } 
                        if (offY < y[c])
                        {
                            exOct += 2;
                        } 
                        if (offZ < z[c]) 
                        {
                            exOct += 4;
                        }
                        parent[c] = cell;
                        child[NUMBER_OF_CELLS * cell + exOct] = c;

                        node = cell;

                        childPath = 0;

                        if (x[cell] < bodyX)
                        {
                            childPath  = 1;
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

                    parent[bodyIdx] = node;
                    child[NUMBER_OF_CELLS * node + childPath] = bodyIdx;
                    
                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                    child[locked] = patch;
                }

                localMaxD = max(depth, localMaxD);
                bodyIdx += stepSize;
                newBody = true;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    atom_max(maxDepth, localMaxD);
    
}