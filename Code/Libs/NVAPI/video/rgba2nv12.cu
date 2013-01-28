#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>


texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat>		rgbatex;


__global__
void rgba2nv12_kernel(char* dst, std::size_t dstpitch, int width, int height, int paddedheight)
{
	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	int yid = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xid < width) && (yid < height))
	{
		float4	rgba = tex2D(rgbatex, xid, height - yid - 1);

		// FIXME: no idea *where* the chroma samles are supposed to be
		//        do i need to average the rgba samples and then convert?
		//        or do i just subsample?
		// FIXME: pick average from mipmap


		float	y = 0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z;
		float   u = (rgba.z - y) * 0.565f;
		float   v = (rgba.x - y) * 0.713f;

		u += 0.5f;
		v += 0.5f;

		// notice the multiplication with 255: incomding is normalised texture read but outgoing is global mem write
		dst[yid * dstpitch + xid] = y * 255;

		if (((xid % 2) == 0) && ((yid % 2) == 0))
		{
			dst[paddedheight * dstpitch + yid/2 * dstpitch + xid] = u * 255;
			dst[paddedheight * dstpitch + yid/2 * dstpitch + xid+1] = v * 255;
		}
	}
}

extern "C"
bool rgba2nv12(char* dst, std::size_t dstpitch, cudaArray_t array, int width, int height, int paddedheight)
{
	cudaChannelFormatDesc	c = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaError_t error = cudaBindTextureToArray(&rgbatex, array, &c);
	if (error != cudaSuccess)
		return false;

	const int	TILE_DIM = 16;
	dim3	grid(width / TILE_DIM + 1, height / TILE_DIM + 1);
	dim3	threads(TILE_DIM, TILE_DIM);

	rgba2nv12_kernel<<<grid, threads>>>(dst, dstpitch, width, height, paddedheight);

	return true;
}
