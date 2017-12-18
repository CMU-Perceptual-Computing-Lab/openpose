// #include <openpose/experimental/tracking/pyramidalLK.hpp>



// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <iomanip>
// #include <vector>
// #include <cstdio>
// #include <ctime>
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/gpu/gpu.hpp>

// using namespace std;
// using namespace cv;

// namespace op
// {
//     /* Error codes for kernel caller*/
//     #define IMAGE_SIZES_NEQUAL -1

//     /* Point error status */
//     #define OUT_OF_FRAME 2
//     #define ZERO_DENOMINATOR 3

//     /* Global parameters */
//     int block_size = 128;

//     __global__ void bkernel (float* img,
//                              int w,
//                              int h)
//     {
//         /*2D Index of current thread */
//         const int x = blockIdx.x * blockDim.x + threadIdx.x;
//         const int y = blockIdx.y * blockDim.y + threadIdx.y;

//         img[(y*w) + x]*=1.80;
       
//         if (img[(y*w) + x] > 255.0)
//             img[(y*w) + x] = 255.0;

//     }

//     __global__ void lkpyramid_kernel (float* I, float *J, int w, int h, 
//                                       float2 *ptsI, float2 *ptsJ, int npoints,
//                                       char *status, int patch_size, float scale)
//     {
//         /*2D Index of current thread */
//         const int idx = blockIdx.x * blockDim.x + threadIdx.x;

//         if (idx >= npoints || status[idx] > 0)
//             return;

//         if (scale != 1.0)
//         {
//             ptsI[idx].x *= scale;
//             ptsI[idx].y *= scale;
//             ptsJ[idx].x *= scale;
//             ptsJ[idx].y *= scale;
//         }

//         /* Get frame I and frame J coordinates */
//         float xi = ptsI[idx].x;
//         float yi = ptsI[idx].y;
//         float xj = ptsJ[idx].x;
//         float yj = ptsJ[idx].y;
        
//         /* Validate patch area, +-1 up/down left/right required
//            for x and y gradient.
//         */
//         if ((int)xi-1 < 0 || (int) yi-1 < 0 || (int)xi+1 >= w ||  (int)yi+1 >= h)
//         {
//             status[idx] = OUT_OF_FRAME;
//             return;
//         } 
//         /* Validate patch area for J */
//         if ((int)xj < 0 || (int)yj < 0 || (int)xj >= w ||(int)yj >= h)  
//         {
//             status[idx] = OUT_OF_FRAME;
//             return;
//         }

//         /* Sum terms to calculate delta_u and delta_v */    
//         float sum_xx = 0.0, sum_yy = 0.0, sum_xt = 0.0, 
//               sum_yt = 0.0, sum_xy = 0.0;

//         /* Radius (r) = floor(patch_size/2) */
//         int r = patch_size / 2;

//         /* Tempral scalars */
//         float dx = 0.0, dy = 0.0, dt = 0.0;

//         /* Acumulate sum over patch */
//         for (int i = -r; i <= r; i++)
//             for (int j = -r; j <= r; j++)
//             {
//                 dx = (I[((int)yi+i)*w + ((int)xi+1+j)] - 
//                       I[((int)yi+i)*w + ((int)xi-1+j)]) / 2.0;    
//                 dy = (I[((int)yi+i+1)*w + ((int)xi+j)] - 
//                       I[((int)yi+i-1)*w + ((int)xi+j)]) / 2.0;
//                 dt = J[((int)yj+i)*w + ((int)xj+j)] - 
//                      I[((int)yi+i)*w + ((int)xi+j)];
//                 sum_xx += dx*dx;
//                 sum_yy += dy*dy;
//                 sum_xy += dx*dy;
//                 sum_yt += dy*dt;
//                 sum_xt += dx*dt; 
//             }

//         /* Calculate displacement in 'x':u and displacement in 'y':x */

//         /* Get numerator and denominator of u and v */
//         float den = (sum_xx*sum_yy) - (sum_xy * sum_xy);

//         if (den == 0.0)
//         {
//             status[idx] = ZERO_DENOMINATOR;
//             return;
//         } 
        
//         float num_u = (-1.0 * sum_yy * sum_xt) + (sum_xy * sum_yt);
//         float num_v = (-1.0 * sum_xx * sum_yt) + (sum_xt * sum_xy);
//         float u = num_u / den;
//         float v = num_v / den;
        
//         ptsJ[idx].x += u;
//         ptsJ[idx].y += v;
//     }

//     /* Given an OpenCV image 'img', build a gaussian pyramid of size 'levels' */
//     void build_gaussian_pyramid_gpu(const Mat& img, int levels, vector<cv::gpu::GpuMat>& pyramid)
//     {
//         cv::gpu::GpuMat current;
//         pyramid.clear();

//         current.upload(img);
//         pyramid.push_back(current);

//         for(int i = 0; i < levels - 1; i++)
//         {
//             cv::gpu::GpuMat tmp;
//             cv::gpu::pyrDown(pyramid[pyramid.size() - 1], tmp);
//             pyramid.push_back(tmp);
//         }
//     }


//     void call_bkernel(cv::gpu::GpuMat img, int w, int h)
//     {
//          Device image 
//         float *img_ptr = (float*) img.ptr<float>();
//         float2 *cpuf2 = (float2 *) malloc(sizeof(float2)*10);
        
//         /* BLock width */
//         int block_width = 16;


//         /* Calculate Grid Size */
//         const dim3 block(block_width, block_width);
//         const dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

//         /* Launch Kernel */
//         bkernel<<<grid,block>>>(img_ptr, w, h);
//         cudaDeviceSynchronize();

//         printf("Finished calling kernel\n");

//     }


//     int lkpyramidal_iteration_gpu (cv::gpu::GpuMat& I, cv::gpu::GpuMat& J, 
//                                     float2 *ptsI,
//                                     float2 *ptsJ, 
//                                     int patch_size,
//                                     int npoints, 
//                                     char *status,
//                                     float scale)
//     { 

//         /* Get float pointers of I and J */
//         float *ptrI = (float *) I.ptr<float>();
//         float *ptrJ = (float *) J.ptr<float>();    

//         /* Validate equal dimension for both images and assign width and height */
//         int w = I.cols;
//         int h = I.rows;
        
//         if (w != J.cols || h != I.rows) return IMAGE_SIZES_NEQUAL;

//         /* Block size and number of blocks */
//         int bsize = block_size;
//         int nblocks = (npoints + bsize -1) / bsize;

//         /* Launch kernel */ 
//         lkpyramid_kernel<<<nblocks,bsize>>>(ptrI,ptrJ,w,h,ptsI,ptsJ,
//                                           npoints,status, patch_size, scale);      
//         /* Wait for all cuda threads to finish */
//         cudaDeviceSynchronize();
        
//         return 0;
//     }

//     int pyramidalLKGpu(vector<Point2f>& ptsI, vector<Point2f>& ptsJ,
//                         vector<char>& status,
//                        const cv::Mat& I, const cv::Mat& J,
//                         const int levels, const int patch_size) 
//     {
//         vector<cv::gpu::GpuMat> pyrI;
//         vector<cv::gpu::GpuMat> pyrJ;

//         /* Allocate ptsJ and initialize */
//         ptsJ.clear();
//         ptsJ.assign(ptsI.begin(), ptsI.end());
        
//         /* Build Gaussian pyramid for both I and J */
//         build_gaussian_pyramid_gpu(I, levels, pyrI);            
//         build_gaussian_pyramid_gpu(J, levels, pyrJ);                                        
//         /* Convert Point2f vector to float2 array */
//         int pts_size = sizeof(float2) * ptsI.size();
//         float2 *ptsI_f2 = (float2 *) malloc(pts_size);
//         float2 *ptsJ_f2 = (float2 *) malloc(pts_size);

//         for (int i = 0; i < ptsI.size(); i++)
//         {
//             ptsI_f2[i].x = ptsI[i].x;
//             ptsI_f2[i].y = ptsI[i].y;
//         }

//         /* Allocate pts on the GPU */
//         float2 *ptsI_gpu, *ptsJ_gpu;
//         cudaMalloc(&ptsI_gpu, pts_size);
//         cudaMalloc(&ptsJ_gpu, pts_size);
//         /* Copy pts CPU -> GPU */
//         cudaMemcpy(ptsI_gpu, ptsI_f2, pts_size, cudaMemcpyHostToDevice);
//         cudaMemcpy(ptsJ_gpu, ptsI_f2, pts_size, cudaMemcpyHostToDevice);
//         /* Move status vector to the gpu */
//         char *status_gpu;
//         cudaMalloc(&status_gpu, status.size());
//         cudaMemcpy(status_gpu, status.data(), status.size(), cudaMemcpyHostToDevice);

//         float scale = 1.0 / (float) (1<<(levels));

//         int npoints = ptsI.size();

//         /* Iterate level by level */
//         for (int l = levels - 1; l >= 0; l--)
//         {
//             scale *= 2.0;
                
//             if (l != levels - 1) scale = 2.0;

//             lkpyramidal_iteration_gpu (pyrI[l], pyrJ[l], ptsI_gpu, ptsJ_gpu,
//                                        patch_size, npoints, status_gpu,scale); 
//         } 

//         /* Copy points and status back */
//         cudaMemcpy(ptsI_f2, ptsI_gpu, pts_size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(ptsJ_f2, ptsJ_gpu, pts_size, cudaMemcpyDeviceToHost);

//         cudaMemcpy(status.data(),status_gpu, status.size(), cudaMemcpyDeviceToHost);

//         /* Recover Point2f I and J */
//         for (int i = 0; i < ptsI.size(); i++)
//         {
//             ptsI[i].x = ptsI_f2[i].x;
//             ptsI[i].y = ptsI_f2[i].y;
//             ptsJ[i].x = ptsJ_f2[i].x;
//             ptsJ[i].y = ptsJ_f2[i].y;
//         }
        
//         /* Free GPU allocated memory */
//         cudaFree(ptsI_gpu);
//         cudaFree(ptsJ_gpu);
//         cudaFree(status_gpu);

//         return 0;
//     }
// }     
