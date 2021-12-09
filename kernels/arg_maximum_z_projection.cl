__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void arg_maximum_z_projection (
    IMAGE_src_TYPE   src,
    IMAGE_dst0_TYPE  dst0,
    IMAGE_dst1_TYPE  dst1,
) 
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
 
  IMAGE_src_PIXEL_TYPE max = 0;
  int arg = 0;
  for(int z = 0; z < GET_IMAGE_DEPTH(src); ++z)
  {
    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x,y,z,0)).x;
    if (value > max || z == 0) {
      max = value;
      arg = z;
    }
  }
  
  WRITE_IMAGE(dst0, POS_dst0_INSTANCE(x,y,0,0), CONVERT_dst0_TYPE(max));
  WRITE_IMAGE(dst1, POS_dst1_INSTANCE(x,y,0,0), CONVERT_dst1_TYPE(arg));
}